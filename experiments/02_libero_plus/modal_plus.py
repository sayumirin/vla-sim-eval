"""Isolated LIBERO-plus image; model cache shared with completed stage one."""
import json
from pathlib import Path
import modal

ROOT = Path(__file__).parent
LEROBOT_COMMIT = '72e1d9e5e76ee719bacfc78bc34d817bee5fb521'
PLUS_COMMIT = '4976dc30028e805ff8094b55501d532c48fec182'
app = modal.App('knightec-vla-stage2')
volume = modal.Volume.from_name('knightec-vla-stage1')
secrets = [modal.Secret.from_name('knightec-huggingface')]
image = (
    modal.Image.debian_slim(python_version='3.12')
    .apt_install('git', 'build-essential', 'ffmpeg', 'libegl1', 'libgl1',
                 'libgles2', 'libgl1-mesa-dri', 'libglib2.0-0', 'libosmesa6')
    .pip_install('cmake>=3.29,<4', 'ninja')
    .run_commands(
        'ln -sf /usr/local/lib/python3.12/site-packages/cmake/data/bin/cmake /usr/local/bin/cmake',
        'git init /opt/lerobot',
        'git -C /opt/lerobot remote add origin https://github.com/huggingface/lerobot.git',
        f'git -C /opt/lerobot fetch --depth 1 origin {LEROBOT_COMMIT}',
        f'git -C /opt/lerobot checkout --detach {LEROBOT_COMMIT}',
    )
    .pip_install('/opt/lerobot[libero,evaluation,pi]', 'torch==2.11.0',
                 'torchvision==0.26.0', 'cmake<4')
    .apt_install('libopengl0')
    .env({'HF_HOME': '/data/huggingface', 'LIBERO_CONFIG_PATH': '/tmp/libero-config',
          'MUJOCO_GL': 'egl', 'TOKENIZERS_PARALLELISM': 'false',
          'HF_HUB_DISABLE_TELEMETRY': '1', 'PYTHONUNBUFFERED': '1'})
    .apt_install('libexpat1', 'libfontconfig1-dev', 'libmagickwand-dev')
    .pip_install('robosuite==1.4.1', 'Wand==0.6.13', 'scikit-image==0.25.2', 'gym==0.26.2')
    .run_commands(
        'git init /opt/libero-plus',
        'git -C /opt/libero-plus remote add origin https://github.com/sylvestf/LIBERO-plus.git',
        f'git -C /opt/libero-plus fetch --depth 1 origin {PLUS_COMMIT}',
        f'git -C /opt/libero-plus checkout --detach {PLUS_COMMIT}',
        'pip install --no-deps -e /opt/libero-plus',
        'pip uninstall -y hf-libero',
    )
    .env({'PYTHONPATH': '/opt/libero-plus'})
    .add_local_file(ROOT / 'configs/plus_tasks.json', '/opt/knightec/plus_tasks.json', copy=True)
    .add_local_file(ROOT / 'run_plus_eval.py', '/opt/knightec/run_plus_eval.py', copy=True)
)


def configure():
    import os
    import yaml
    package = Path('/opt/libero-plus/libero/libero')
    assets = package / 'assets'
    if not assets.exists():
        assets.symlink_to('/data/plus/assets', target_is_directory=True)
    config = Path(os.environ['LIBERO_CONFIG_PATH'])
    config.mkdir(exist_ok=True)
    (config / 'config.yaml').write_text(yaml.safe_dump({
        'assets': str(assets), 'bddl_files': str(package / 'bddl_files'),
        'init_states': str(package / 'init_files'), 'datasets': '/data/datasets',
        'benchmark_root': str(package)}))


def bake_assets():
    """Bake exact archive bytes into the image on CPU, avoiding tiny-file network IO."""
    import shutil
    import zipfile
    from huggingface_hub import hf_hub_download
    manifest = json.loads(Path('/opt/knightec/assets_manifest.json').read_text())
    archive = hf_hub_download(manifest['repo'], repo_type='dataset', filename='assets.zip',
                              revision=manifest['revision'])
    target_root = Path('/opt/libero-plus/libero/libero/assets')
    target_root.mkdir(exist_ok=True)
    count = 0
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            parts = Path(item.filename).parts
            if 'assets' not in parts or item.is_dir():
                continue
            relative = Path(*parts[parts.index('assets') + 1:])
            assert not relative.is_absolute() and '..' not in relative.parts
            if relative.parts[0] == 'new_objects' and not any(relative.as_posix().startswith(p) for p in manifest['custom_object_prefixes']):
                continue
            target = target_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(item) as source, target.open('wb') as dest:
                shutil.copyfileobj(source, dest)
            count += 1
    assert count == manifest['extracted_files'], (count, manifest['extracted_files'])
    print(f'Baked {count} exact asset files into the evaluation image.', flush=True)


evaluation_image = (
    image.add_local_file(ROOT / 'audit/assets_manifest.json', '/opt/knightec/assets_manifest.json', copy=True)
    .run_function(bake_assets, volumes={'/data': volume}, secrets=secrets,
                  cpu=2, memory=8192, timeout=1200)
)


@app.function(image=image, volumes={'/data': volume}, cpu=2, memory=8192,
              secrets=secrets, timeout=2400, max_containers=1, retries=0)
def prepare():
    import ast
    import hashlib
    import re
    import shutil
    import zipfile
    from huggingface_hub import HfApi, hf_hub_download
    volume.reload()
    root = Path('/data/plus')
    root.mkdir(exist_ok=True)
    manifest = root / 'assets_manifest.json'
    if not manifest.exists():
        package = Path('/opt/libero-plus/libero/libero')
        config = json.loads(Path('/opt/knightec/plus_tasks.json').read_text())
        # The archive contains >400,000 meshes for 1,000+ optional objects.
        # Extract complete asset subtrees for every custom object referenced
        # by our selected BDDL files, plus all ordinary/shared scene assets.
        needed_types = set()
        for category in config['categories'].values():
            for row in category['tasks']:
                bddl = package / 'bddl_files/libero_goal' / (row['name'].split('_view_')[0] + '.bddl')
                needed_types.update(re.findall(r'-\s+([a-zA-Z_0-9]+)', bddl.read_text()))
        custom_paths = {}
        tree = ast.parse((package / 'envs/objects/custom_objects.py').read_text())
        for cls in tree.body:
            if not isinstance(cls, ast.ClassDef):
                continue
            for method in cls.body:
                if not isinstance(method, ast.FunctionDef) or method.name != '__init__':
                    continue
                defaults = [n.value for n in method.args.defaults if isinstance(n, ast.Constant) and isinstance(n.value, str)]
                paths = [n.value for n in ast.walk(method) if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith('assets/new_objects/')]
                for name in defaults:
                    if paths:
                        custom_paths[name] = '/'.join(paths[0].split('/')[1:4]) + '/'
        unknown = {name for name in needed_types if re.search(r'__\d+$', name) and name not in custom_paths}
        assert not unknown, unknown
        required_prefixes = sorted({custom_paths[name] for name in needed_types if name in custom_paths})
        revision = HfApi().dataset_info('Sylvest/LIBERO-plus').sha
        archive = hf_hub_download('Sylvest/LIBERO-plus', repo_type='dataset',
                                  filename='assets.zip', revision=revision)
        count = 0
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                parts = Path(item.filename).parts
                if 'assets' not in parts or item.is_dir():
                    continue
                relative = Path(*parts[parts.index('assets') + 1:])
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Unsafe archive path')
                if relative.parts[0] == 'new_objects' and not any(relative.as_posix().startswith(prefix) for prefix in required_prefixes):
                    continue
                target = root / 'assets' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.stat().st_size == item.file_size:
                    count += 1
                    continue
                with z.open(item) as source, target.open('wb') as dest:
                    shutil.copyfileobj(source, dest)
                count += 1
        assert count > 100
        manifest.write_text(json.dumps({'repo': 'Sylvest/LIBERO-plus',
            'revision': revision, 'archive_sha256': hashlib.file_digest(open(archive, 'rb'), 'sha256').hexdigest(),
            'extracted_files': count, 'selection': 'all shared assets plus complete selected custom-object subtrees',
            'custom_object_prefixes': required_prefixes}, indent=2))
        volume.commit()
    configure()
    import contextlib
    import io
    from libero.libero import benchmark
    from lerobot.envs.libero import get_task_init_states
    config = json.loads(Path('/opt/knightec/plus_tasks.json').read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        suite = benchmark.get_benchmark_dict()['libero_goal'](task_order_index=0)
    audit = []
    for category in config['categories'].values():
        for row in category['tasks']:
            task = suite.get_task(row['task_id'])
            assert task.name == row['name']
            states = get_task_init_states(suite, row['task_id'], is_libero_plus=True)
            bddl = Path('/opt/libero-plus/libero/libero/bddl_files/libero_goal') / (task.name.split('_view_')[0] + '.bddl')
            assert bddl.is_file(), bddl
            audit.append({'task_id': row['task_id'], 'name': task.name, 'category': row['category'],
                          'raw_prompt': task.language, 'prompt': row['prompt'],
                          'init_state_shape': list(states.shape), 'bddl_path': str(bddl)})
    (root / 'selection_audit.json').write_text(json.dumps(audit, indent=2))
    (root / 'plus_tasks.json').write_text(json.dumps(config, indent=2))
    volume.commit()
    return {'assets': json.loads(manifest.read_text()), 'validated_variants': len(audit),
            'libero_plus_commit': PLUS_COMMIT}


@app.function(image=evaluation_image, volumes={'/data': volume}, gpu='A100-40GB',
              cpu=(4, 4), memory=(32768, 32768), timeout=4000,
              secrets=secrets, max_containers=1, scaledown_window=2, retries=0)
def evaluate(category: str, smoke: bool, max_seconds: int):
    import datetime
    import hashlib
    import signal
    import subprocess
    import sys
    import time
    import os
    import torch
    assert category in {'light', 'camera', 'layout'} and 60 <= max_seconds <= 3600
    volume.reload()
    configure()
    artifacts = json.loads(Path('/data/artifacts.json').read_text())
    config = json.loads(Path('/opt/knightec/plus_tasks.json').read_text())
    selected = config['categories'][category]
    ids = selected['task_ids'][:1] if smoke else selected['task_ids']
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = Path('/data/results_plus') / f'{category}_{"smoke" if smoke else "100"}_seed0_{stamp}'
    output.mkdir(parents=True)
    run = {**artifacts, 'libero_plus_commit': PLUS_COMMIT,
           'plus_assets': json.loads(Path('/data/plus/assets_manifest.json').read_text()),
           'category': category, 'smoke': smoke, 'seed': 0, 'task_ids': ids,
           'selection': selected, 'prompt_protocol': config['prompt_protocol'],
           'runner_sha256': hashlib.sha256(Path('/opt/knightec/run_plus_eval.py').read_bytes()).hexdigest(),
           'selection_sha256': hashlib.sha256(Path('/opt/knightec/plus_tasks.json').read_bytes()).hexdigest(),
           'asset_storage': 'Exact archive bytes baked into evaluation image on CPU',
           'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
           'cuda': torch.version.cuda, 'status': 'running', 'output': str(output),
           'max_seconds': max_seconds}
    command = [sys.executable, '/opt/knightec/run_plus_eval.py',
        f'--policy.path={artifacts["model_path"]}', '--policy.device=cuda',
        '--policy.dtype=bfloat16', '--policy.compile_model=false', '--policy.n_action_steps=10',
        '--env.type=libero_plus', '--env.task=libero_goal',
        f'--env.task_ids={json.dumps(ids)}', '--env.control_mode=relative',
        '--env.init_states=true', '--env.hard_reset=true', '--env.max_parallel_tasks=1',
        '--eval.batch_size=1', '--eval.n_episodes=1', '--seed=0', f'--output_dir={output}']
    run['command'] = command
    (output / 'manifest.json').write_text(json.dumps(run, indent=2))
    (output / 'packages.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    started = time.monotonic()
    try:
        with (output / 'eval.log').open('w') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                     env={**os.environ, 'KNIGHTEC_OUTPUT': str(output),
                                          'KNIGHTEC_SMOKE': '1' if smoke else '0'})
            try:
                code = child.wait(timeout=max_seconds)
                run.update(status='completed' if code == 0 else 'failed', exit_code=code)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                run['status'] = 'timed_out'
        if (output / 'eval_info.json').exists():
            run['results'] = json.loads((output / 'eval_info.json').read_text())
        run['log_tail'] = (output / 'eval.log').read_text()[-12000:]
    finally:
        run['wall_seconds'] = time.monotonic() - started
        (output / 'manifest.json').write_text(json.dumps(run, indent=2))
        volume.commit()
    return run


@app.local_entrypoint()
def main(mode: str = 'prepare', category: str = 'all', max_seconds: int = 3600):
    assert mode in {'prepare', 'smoke', 'full'}
    if mode == 'prepare':
        result = prepare.remote()
        (ROOT / 'prepare_result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        return
    order = ['layout', 'light', 'camera'] if mode == 'full' else ['light', 'camera', 'layout']
    cats = order if category == 'all' else [category]
    for cat in cats:
        result = evaluate.remote(cat, mode == 'smoke', max_seconds)
        dest = ROOT / 'modal_results'
        dest.mkdir(exist_ok=True)
        (dest / f'{Path(result["output"]).name}.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k,v in result.items() if k not in {'selection', 'log_tail', 'results'}}, indent=2), flush=True)
        print(json.dumps(result.get('results', {}).get('overall', {})), flush=True)
        if result['status'] != 'completed':
            print(result['log_tail'], flush=True)
            raise RuntimeError(result['status'])
