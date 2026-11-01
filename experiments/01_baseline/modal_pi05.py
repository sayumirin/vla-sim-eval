"""Pinned stage-one LIBERO evaluation on Modal. Importing does not start compute.

modal run modal_pi05.py --mode smoke --max-seconds 1200
modal run modal_pi05.py --mode goal --seed 0 --max-seconds 3600
modal run modal_pi05.py --mode baseline --seed 0 --max-seconds 7200
Each baseline invocation is one seed, four suites, 400 episodes.
"""

import json
from pathlib import Path

import modal

LEROBOT_COMMIT = '72e1d9e5e76ee719bacfc78bc34d817bee5fb521'
MODEL_REPO = 'lerobot/pi05_libero_finetuned_v044'
MODEL_REVISION = '7152f13e34c6c115d808aea0e5d08aca74eac14b'

app = modal.App('knightec-vla-stage1')
volume = modal.Volume.from_name('knightec-vla-stage1', create_if_missing=True)
hf_secrets = [modal.Secret.from_name('knightec-huggingface')]
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
)


def configure_libero(assets: str) -> None:
    import importlib.util
    import os
    import yaml

    package = Path(importlib.util.find_spec('libero').origin).parent / 'libero'
    config_dir = Path(os.environ['LIBERO_CONFIG_PATH'])
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / 'config.yaml').write_text(yaml.safe_dump({
        'assets': assets, 'bddl_files': str(package / 'bddl_files'),
        'init_states': str(package / 'init_files'),
        'datasets': '/data/datasets', 'benchmark_root': str(package),
    }))


@app.function(image=image, volumes={'/data': volume}, cpu=2, memory=8192,
              secrets=hf_secrets, timeout=1800, max_containers=1, retries=0)
def prepare() -> dict:
    """Download on CPU so model downloads do not occupy a billed A100."""
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer

    manifest_path = Path('/data/artifacts.json')
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    assets_revision = HfApi().dataset_info('lerobot/libero-assets').sha
    assets = snapshot_download('lerobot/libero-assets', repo_type='dataset',
                               revision=assets_revision)
    model = snapshot_download(MODEL_REPO, revision=MODEL_REVISION)
    processor = json.loads((Path(model) / 'policy_preprocessor.json').read_text())
    for step in processor['steps']:
        tokenizer = step.get('config', {}).get('tokenizer_name')
        if tokenizer:
            AutoTokenizer.from_pretrained(tokenizer)
    manifest = {'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION,
                'lerobot_commit': LEROBOT_COMMIT, 'model_path': model,
                'assets_path': assets, 'assets_revision': assets_revision}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    volume.commit()
    return manifest


@app.function(image=image, volumes={'/data': volume}, gpu='A100-40GB',
              cpu=(4, 4), memory=(32768, 32768), timeout=7500,
              secrets=hf_secrets, max_containers=1, scaledown_window=2, retries=0)
def evaluate(mode: str, seed: int, max_seconds: int) -> dict:
    import datetime
    import os
    import subprocess
    import sys
    import time
    import torch

    if mode not in {'smoke', 'goal', 'baseline'} or not 60 <= max_seconds <= 7200:
        raise ValueError('Invalid mode or runtime limit')
    volume.reload()
    manifest = json.loads(Path('/data/artifacts.json').read_text())
    configure_libero(manifest['assets_path'])
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = Path('/data/results') / f'{mode}_seed{seed}_{stamp}'
    output.mkdir(parents=True)
    run = {**manifest, 'mode': mode, 'seed': seed, 'max_seconds': max_seconds,
           'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
           'cuda': torch.version.cuda, 'compile_model': False,
           'status': 'running', 'output': str(output)}
    (output / 'manifest.json').write_text(json.dumps(run, indent=2))
    (output / 'packages.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True))
    command = [
        'lerobot-eval', f'--policy.path={manifest["model_path"]}',
        '--policy.device=cuda', '--policy.dtype=bfloat16',
        '--policy.compile_model=false', '--policy.n_action_steps=10',
        '--env.type=libero', '--env.control_mode=relative',
        '--env.init_states=true', '--env.hard_reset=true',
        '--env.max_parallel_tasks=1', '--eval.batch_size=1',
        f'--seed={seed}', f'--output_dir={output}',
    ]
    if mode == 'smoke':
        command += ['--env.task=libero_object', '--env.task_ids=[0]', '--eval.n_episodes=2']
    elif mode == 'goal':
        command += ['--env.task=libero_goal', '--eval.n_episodes=10']
    else:
        command += ['--env.task=libero_spatial,libero_object,libero_goal,libero_10',
                    '--eval.n_episodes=10']
    run['command'] = command
    started = time.monotonic()
    try:
        with (output / 'eval.log').open('w') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            try:
                code = child.wait(timeout=max_seconds)
                run['status'] = 'completed' if code == 0 else 'failed'
                run['exit_code'] = code
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                run['status'] = 'timed_out'
        result_file = output / 'eval_info.json'
        if result_file.exists():
            run['results'] = json.loads(result_file.read_text())
        run['log_tail'] = (output / 'eval.log').read_text()[-16000:]
    finally:
        run['wall_seconds'] = time.monotonic() - started
        (output / 'manifest.json').write_text(json.dumps(run, indent=2))
        volume.commit()
    return run


@app.local_entrypoint()
def main(mode: str = 'smoke', seed: int = 0, max_seconds: int = 1200):
    if mode not in {'smoke', 'goal', 'baseline'} or not 60 <= max_seconds <= 7200:
        raise ValueError('Use smoke, goal or baseline, with a 60–7200 second timeout')
    print(json.dumps(prepare.remote(), indent=2), flush=True)
    result = evaluate.remote(mode, seed, max_seconds)
    destination = Path(__file__).parent / 'modal_results'
    destination.mkdir(exist_ok=True)
    (destination / f'{Path(result["output"]).name}.json').write_text(
        json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if result['status'] != 'completed':
        raise RuntimeError(f'Evaluation {result["status"]}; see saved logs')
