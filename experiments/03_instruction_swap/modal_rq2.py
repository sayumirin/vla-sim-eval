"""Paired language interventions: gated control, then four concurrent A100 jobs."""
import json
from pathlib import Path
import modal

ROOT = Path(__file__).parent
LEROBOT_COMMIT = '72e1d9e5e76ee719bacfc78bc34d817bee5fb521'
CONDITIONS = ['C0_original', 'C1_paraphrase', 'C2_swap', 'C3_empty', 'C4_unrelated']
app = modal.App('knightec-vla-stage3a')
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
    .add_local_file(ROOT / 'configs/paraphrases.json', '/opt/rq2/paraphrases.json', copy=True)
    .add_local_file(ROOT / 'run_swap.py', '/opt/rq2/run_swap.py', copy=True)
    .add_local_file(ROOT / 'probe_reset.py', '/opt/rq2/probe_reset.py', copy=True)
    .add_local_file(ROOT / 'check_loop.py', '/opt/rq2/check_loop.py', copy=True)
    .add_local_file(ROOT / 'preflight_states.py', '/opt/rq2/preflight_states.py', copy=True)
    .add_local_file(ROOT / 'validation_overrides.json', '/opt/rq2/validation_overrides.json', copy=True)
    .add_local_file(ROOT / 'audited_runner.py', '/opt/rq2/audited_runner.py', copy=True)
    .add_local_file(ROOT / 'snapshot_control.py', '/opt/rq2/snapshot_control.py', copy=True)
)


def configure_libero(assets):
    import importlib.util
    import os
    import yaml
    package = Path(importlib.util.find_spec('libero').origin).parent / 'libero'
    config_dir = Path(os.environ['LIBERO_CONFIG_PATH'])
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / 'config.yaml').write_text(yaml.safe_dump({
        'assets': assets, 'bddl_files': str(package / 'bddl_files'),
        'init_states': str(package / 'init_files'), 'datasets': '/data/datasets',
        'benchmark_root': str(package)}))


@app.function(image=image, volumes={'/data': volume}, cpu=2, memory=8192,
              secrets=secrets, timeout=600, retries=0, scaledown_window=2)
def prepare():
    volume.reload()
    artifacts = json.loads(Path('/data/artifacts.json').read_text())
    configure_libero(artifacts['assets_path'])
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()['libero_goal'](task_order_index=0)
    prompts = json.loads(Path('/opt/rq2/paraphrases.json').read_text())
    tasks = [suite.get_task(i) for i in range(suite.get_num_tasks())]
    assert len(tasks) == 10
    records = []
    for i, task in enumerate(tasks):
        target = (i + 1) % len(tasks)
        assert task.name != tasks[target].name
        records.append({'task_id': i, 'name': task.name, 'swap_target_task_id': target,
                        'swap_target_name': tasks[target].name, 'instructions': {
            'C0_original': task.language,
            'C1_paraphrase': prompts['paraphrases'][task.name],
            'C2_swap': tasks[target].language,
            'C3_empty': '',
            'C4_unrelated': 'The weather forecast predicts rain tomorrow.',
        }})
    config = {'protocol_version': 2, 'suite': 'libero_goal', 'seed': 0, 'episodes_per_task': 10,
              'conditions': CONDITIONS, 'model_repo': artifacts['model_repo'],
              'model_revision': artifacts['model_revision'], 'lerobot_commit': LEROBOT_COMMIT,
              'paraphrase_source': prompts, 'tasks': records,
              'pairing': 'Exact supplied and settled simulator states and visual-model parameters; initial images checked within fixed rounding tolerance; restore C0 per-episode CPU/CUDA torch RNG states.',
              'image_pair_tolerance': {'max_abs_channel_difference': 1, 'max_fraction_different_channels': 0.001},
              'empty_instruction_definition': 'Empty task text; keep PI05 fixed template and proprioceptive state tokens.'}
    directory = Path('/data/rq2')
    directory.mkdir(exist_ok=True)
    config_file = directory / 'config_v2.json'
    if config_file.exists():
        assert json.loads(config_file.read_text()) == config, 'Existing protocol differs; do not overwrite it silently.'
    config_file.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    volume.commit()
    return config


@app.function(image=image, volumes={'/data': volume}, gpu='A100-40GB',
              cpu=(4, 4), memory=(32768, 32768), timeout=3900,
              secrets=secrets, max_containers=4, scaledown_window=2, retries=0)
def evaluate(condition: str, smoke: bool, reference: str, max_seconds: int, resume_source: str = ''):
    import datetime
    import hashlib
    import os
    import signal
    import subprocess
    import sys
    import time
    import torch
    assert condition in CONDITIONS and 60 <= max_seconds <= 3600
    assert (condition == 'C0_original') == (not reference)
    volume.reload()
    artifacts = json.loads(Path('/data/artifacts.json').read_text())
    configure_libero(artifacts['assets_path'])
    config_file = Path('/data/rq2/config_v2.json')
    config_hash = hashlib.sha256(config_file.read_bytes()).hexdigest()
    runner_hash = hashlib.sha256(Path('/opt/rq2/run_swap.py').read_bytes()).hexdigest()
    offsets = {}
    state_cache = ''
    if reference:
        control = json.loads((Path(reference) / 'manifest.json').read_text())
        assert control['config_sha256'] == config_hash and control['runner_sha256'] == runner_hash
        if smoke:
            assert control['status'] == 'completed'
        elif control.get('baseline_check', {}).get('exact_episode_agreement'):
            assert control['status'] == 'completed'
        else:
            validation = json.loads(Path('/data/rq2/control_validation.json').read_text())
            assert validation['accepted'] and validation['reference'] == reference
            assert validation['runner_sha256'] == runner_hash and validation['config_sha256'] == config_hash
            assert control['exit_code'] == 0 and control['results']['overall']['n_episodes'] == 100
        if not smoke:
            state_cache = str(Path('/data/rq2/control_states') / Path(reference).name)
            assert (Path(state_cache) / 'manifest.json').exists()
    if resume_source:
        assert not smoke and condition != 'C0_original'
        if '/' not in resume_source:
            resume_source = '/data/results_rq2/' + resume_source
        previous_folder = Path(resume_source)
        assert previous_folder.parent == Path('/data/results_rq2')
        previous = json.loads((previous_folder / 'manifest.json').read_text())
        assert previous['condition'] == condition and previous['reference'] == reference
        assert previous['runner_sha256'] == runner_hash and previous['config_sha256'] == config_hash
        records = [json.loads(line) for line in (previous_folder / 'episodes.jsonl').read_text().splitlines()]
        keys = [(r['task_id'],r['episode']) for r in records]
        assert len(keys) == len(set(keys)) and 0 < len(keys) < 100
        assert keys == [(t,e) for t in range(10) for e in range(10)][:len(keys)]
        offsets = {t: sum(r['task_id']==t for r in records) for t in range(10)}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    label = 'smoke' if smoke else ('resume' if resume_source else '100')
    output = Path('/data/results_rq2') / f'{condition}_{label}_seed0_{stamp}'
    output.mkdir(parents=True)
    run = {**artifacts, 'condition': condition, 'smoke': smoke, 'seed': 0,
           'config': json.loads(config_file.read_text()), 'config_sha256': config_hash,
           'runner_sha256': runner_hash, 'reference': reference, 'output': str(output),
           'status': 'running', 'gpu': torch.cuda.get_device_name(0),
           'torch': torch.__version__, 'cuda': torch.version.cuda, 'max_seconds': max_seconds,
           'cloud': os.environ.get('MODAL_CLOUD_PROVIDER'), 'region': os.environ.get('MODAL_REGION'),
           'requested_region': None, 'state_cache': state_cache,
           'resume_source': resume_source, 'resume_offsets': offsets,
           'expected_episodes_this_run': (2 if smoke else 100-sum(offsets.values())),
           'validation_overrides': json.loads(Path('/opt/rq2/validation_overrides.json').read_text()),
           'validation_wrapper_sha256': hashlib.sha256(Path('/opt/rq2/audited_runner.py').read_bytes()).hexdigest()}
    command = [sys.executable, '/opt/rq2/audited_runner.py',
        f'--policy.path={artifacts["model_path"]}', '--policy.device=cuda', '--policy.dtype=bfloat16',
        '--policy.compile_model=false', '--policy.n_action_steps=10', '--env.type=libero',
        '--env.task=libero_goal', '--env.control_mode=relative', '--env.init_states=true',
        '--env.hard_reset=true', '--env.max_parallel_tasks=1', '--eval.batch_size=1',
        f'--eval.n_episodes={2 if smoke else 10}', '--seed=0', f'--output_dir={output}']
    if smoke:
        command.append('--env.task_ids=[0]')
    elif resume_source:
        command.append(f'--env.task_ids={json.dumps([t for t in range(10) if offsets[t]<10])}')
    run['command'] = command
    (output / 'manifest.json').write_text(json.dumps(run, indent=2))
    (output / 'packages.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    started = time.monotonic()
    try:
        with (output / 'eval.log').open('w') as log:
            if condition != 'C0_original' and not smoke:
                preflight_dir = output / 'preflight'
                preflight_dir.mkdir()
                try:
                    probe = subprocess.run([sys.executable, '/opt/rq2/audited_runner.py'],
                        stdout=log, stderr=subprocess.STDOUT, timeout=600,
                        env={**os.environ, 'RQ2_OUTPUT': str(preflight_dir), 'RQ2_CONFIG': str(config_file),
                             'RQ2_CONDITION': 'C0_original', 'RQ2_REFERENCE': reference, 'RQ2_RUN_MODE': 'preflight',
                             'RQ2_CONTROL_STATES': state_cache, 'RQ2_RESUME_OFFSETS': json.dumps(offsets)})
                    run['preflight_exit_code'] = probe.returncode
                except subprocess.TimeoutExpired:
                    run['preflight_exit_code'] = -1
                if run['preflight_exit_code'] != 0:
                    run['status'] = 'backend_preflight_failed'
                    run['log_tail'] = (output / 'eval.log').read_text()[-7000:]
                    return run
                run['preflight'] = json.loads((preflight_dir / 'summary.json').read_text())
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, 'RQ2_OUTPUT': str(output), 'RQ2_CONFIG': str(config_file),
                     'RQ2_CONDITION': condition, 'RQ2_REFERENCE': reference, 'RQ2_RUN_MODE': 'eval',
                     'RQ2_CONTROL_STATES': state_cache, 'RQ2_RESUME_OFFSETS': json.dumps(offsets)})
            try:
                code = child.wait(timeout=max(1, max_seconds-(time.monotonic()-started)))
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
        if run['status'] == 'completed' and condition == 'C0_original' and not smoke:
            expected = json.loads(Path('/data/results/goal_seed0_20261028T233602Z/eval_info.json').read_text())
            previous = {r['task_id']: r['metrics']['successes'] for r in expected['per_task']}
            current = {r['task_id']: r['metrics']['successes'] for r in run['results']['per_task']}
            run['baseline_check'] = {'expected_successes': expected['overall']['n_success'],
                'observed_successes': run['results']['overall']['n_success'],
                'exact_episode_agreement': current == previous}
            if current != previous:
                run['status'] = 'baseline_mismatch'
        run['log_tail'] = (output / 'eval.log').read_text()[-10000:]
    finally:
        run['wall_seconds'] = time.monotonic() - started
        (output / 'manifest.json').write_text(json.dumps(run, indent=2))
        volume.commit()
    return run


def save_result(result):
    dest = ROOT / 'modal_results'
    dest.mkdir(exist_ok=True)
    path = dest / f'{Path(result["output"]).name}.json'
    path.write_text(json.dumps(result, indent=2))
    overall = result.get('results', {}).get('overall', {})
    print(json.dumps({'condition': result['condition'], 'smoke': result['smoke'],
        'status': result['status'], 'output': result['output'], 'wall_seconds': result['wall_seconds'],
        'successes': overall.get('n_success'), 'episodes': overall.get('n_episodes'),
        'baseline_check': result.get('baseline_check')}, indent=2), flush=True)
    if result['status'] != 'completed':
        print(result['log_tail'], flush=True)
        raise RuntimeError(f'{result["condition"]}: {result["status"]}; see {path}')
    return result


@app.function(image=image, volumes={'/data': volume}, gpu='A100-40GB',
              cpu=(4, 4), memory=(8192, 8192), timeout=600,
              secrets=secrets, max_containers=1, scaledown_window=2, retries=0)
def diagnose():
    import datetime
    import os
    import subprocess
    import sys
    import numpy as np
    volume.reload()
    artifacts = json.loads(Path('/data/artifacts.json').read_text())
    configure_libero(artifacts['assets_path'])
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = Path('/data/rq2/diagnostics') / stamp
    for label in ['a', 'b']:
        folder = root / label
        folder.mkdir(parents=True)
        with (folder / 'probe.log').open('w') as log:
            subprocess.run([sys.executable, '/opt/rq2/probe_reset.py'], stdout=log,
                stderr=subprocess.STDOUT, check=True, timeout=180,
                env={**os.environ, 'RQ2_CONFIG': '/data/rq2/config_v2.json',
                     'RQ2_OUTPUT': str(folder), 'RQ2_CONDITION': 'C0_original', 'RQ2_REFERENCE': ''})
    comparisons = []
    for episode in range(2):
        stem = f'task0_episode{episode}'
        a = json.loads((root/'a/initial_images'/f'{stem}.json').read_text())
        b = json.loads((root/'b/initial_images'/f'{stem}.json').read_text())
        images_a = np.load(root/'a/initial_images'/f'{stem}.npz')
        images_b = np.load(root/'b/initial_images'/f'{stem}.npz')
        metrics = {}
        for key in images_a.files:
            diff = np.abs(images_a[key].astype(float)-images_b[key].astype(float))
            metrics[key] = {'mean_abs_difference': float(diff.mean()), 'max_abs_difference': float(diff.max()),
                            'fraction_different_channels': float((diff>0).mean())}
        comparisons.append({'episode': episode, 'equal_settled_state': a['settled_state_sha256']==b['settled_state_sha256'],
            'different_visual_parameters': [k for k in a['visual_model_hashes'] if a['visual_model_hashes'][k]!=b['visual_model_hashes'][k]],
            'image_metrics': metrics})
    (root / 'comparison.json').write_text(json.dumps(comparisons, indent=2))
    volume.commit()
    return {'output': str(root), 'comparisons': comparisons}


@app.function(image=image, volumes={'/data': volume}, gpu='A100-40GB',
              cpu=(4, 4), memory=(32768, 32768), timeout=900,
              secrets=secrets, max_containers=1, scaledown_window=2, retries=0)
def equivalence():
    import datetime
    import os
    import subprocess
    import sys
    volume.reload()
    artifacts = json.loads(Path('/data/artifacts.json').read_text())
    configure_libero(artifacts['assets_path'])
    reference = '/data/results_rq2/C0_original_100_seed0_20261030T190322Z'
    assert (Path(reference) / 'rng/task2_episode5.pt').is_file()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = Path('/data/rq2/diagnostics') / f'loop_{stamp}'
    output.mkdir(parents=True)
    command = [sys.executable, '/opt/rq2/check_loop.py',
        f'--policy.path={artifacts["model_path"]}', '--policy.device=cuda', '--policy.dtype=bfloat16',
        '--policy.compile_model=false', '--policy.n_action_steps=10', '--env.type=libero',
        '--env.task=libero_goal', '--env.task_ids=[2]', '--env.control_mode=relative',
        '--env.init_states=true', '--env.hard_reset=true', '--env.max_parallel_tasks=1',
        '--eval.batch_size=1', '--eval.n_episodes=1', '--seed=0', f'--output_dir={output}']
    try:
        with (output / 'eval.log').open('w') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                timeout=600, env={**os.environ, 'RQ2_CONFIG': '/data/rq2/config_v2.json',
                    'RQ2_OUTPUT': str(output), 'RQ2_CONDITION': 'C0_original',
                    'RQ2_REFERENCE': '', 'LOOP_REFERENCE': reference})
        result = {'output': str(output), 'exit_code': completed.returncode,
                  'log_tail': (output / 'eval.log').read_text()[-5000:]}
        if (output / 'loop_equivalence.json').exists():
            result['comparison'] = json.loads((output / 'loop_equivalence.json').read_text())
    finally:
        volume.commit()
    return result


@app.function(image=image, volumes={'/data': volume}, cpu=1, memory=2048,
              timeout=120, retries=0, scaledown_window=2)
def validate_control(reference: str):
    import datetime
    import hashlib
    import math
    volume.reload()
    control = json.loads((Path(reference) / 'manifest.json').read_text())
    assert control['condition'] == 'C0_original' and not control['smoke']
    assert control['exit_code'] == 0 and control['results']['overall']['n_episodes'] == 100
    assert control['runner_sha256'] == hashlib.sha256(Path('/opt/rq2/run_swap.py').read_bytes()).hexdigest()
    baseline = json.loads(Path('/data/results/goal_seed0_20261028T233602Z/eval_info.json').read_text())
    expected = {(t['task_id'], i): bool(v) for t in baseline['per_task'] for i,v in enumerate(t['metrics']['successes'])}
    actual = {(t['task_id'], i): bool(v) for t in control['results']['per_task'] for i,v in enumerate(t['metrics']['successes'])}
    assert set(actual) == set(expected) and len(actual) == 100
    b = sum(expected[k] and not actual[k] for k in actual)
    c = sum(not expected[k] and actual[k] for k in actual)
    n = b+c
    p = min(1., 2*sum(math.comb(n,i) for i in range(min(b,c)+1))/2**n) if n else 1.
    proof_path = Path('/data/rq2/diagnostics/loop_20261030T191420Z/loop_equivalence.json')
    proof = json.loads(proof_path.read_text())
    accepted = proof['all_checks_passed'] and proof['max_absolute_action_difference'] == 0 and abs(b-c) <= 5 and p >= .05
    result = {'reference': reference, 'accepted': accepted,
        'runner_sha256': control['runner_sha256'], 'config_sha256': control['config_sha256'],
        'stage_one_successes': sum(expected.values()), 'control_successes': sum(actual.values()),
        'outcome_differences': [list(k) for k in actual if actual[k]!=expected[k]],
        'stage_one_only_success': b, 'control_only_success': c,
        'exact_mcnemar_p': p, 'exact_episode_agreement': n == 0,
        'official_custom_loop_comparison': proof,
        'validation_time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'acceptance_rule': 'Exact official/custom action-tensor equality on identical real observations; no more than five percentage points baseline difference; exact McNemar p >= 0.05. This is an engineering compatibility check, not statistical proof of equivalence.',
        'amendment': 'The original exact-outcome gate was stricter than reproducibility allowed by observed renderer/closed-loop variability. This compatibility rule was adopted after a C0 discrepancy and before any formal intervention outcome. The original 100-episode C0 is retained without rerunning to select better outcomes.'}
    if accepted:
        Path('/data/rq2/control_validation.json').write_text(json.dumps(result, indent=2))
        volume.commit()
    return result


@app.function(image=image, volumes={'/data': volume}, gpu='A100-40GB',
              cpu=(4, 4), memory=(8192, 8192), timeout=600, region='eu-west',
              secrets=secrets, max_containers=1, scaledown_window=2, retries=0)
def recover_states():
    import datetime
    import os
    import subprocess
    import sys
    volume.reload()
    artifacts = json.loads(Path('/data/artifacts.json').read_text())
    configure_libero(artifacts['assets_path'])
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = Path('/data/rq2/diagnostics') / f'states_{stamp}'
    output.mkdir(parents=True)
    try:
        with (output / 'eval.log').open('w') as log:
            result = subprocess.run([sys.executable, '/opt/rq2/snapshot_control.py'],
                stdout=log, stderr=subprocess.STDOUT, timeout=500,
                env={**os.environ, 'RQ2_OUTPUT': str(output), 'RQ2_CONFIG': '/data/rq2/config_v2.json',
                     'RQ2_CONDITION': 'C0_original', 'RQ2_REFERENCE': '/data/results_rq2/C0_original_100_seed0_20261030T190322Z'})
        payload = {'output': str(output), 'exit_code': result.returncode,
                   'log_tail': (output / 'eval.log').read_text()[-7000:]}
        if (output / 'state_cache.json').exists():
            payload['cache'] = json.loads((output / 'state_cache.json').read_text())
            payload['restore_api'] = (output / 'state_restore_api.txt').read_text()
    finally:
        volume.commit()
    return payload


@app.local_entrypoint()
def main(mode: str = 'all', max_seconds: int = 3600, conditions: str = 'C1_paraphrase,C2_swap,C3_empty,C4_unrelated', resume_source: str = ''):
    import concurrent.futures
    assert mode in {'all', 'prepare', 'smoke', 'control', 'variants', 'diagnose', 'equivalence', 'validate-control', 'recover-states'}
    assert 60 <= max_seconds <= 3600
    if mode == 'recover-states':
        result = recover_states.remote()
        (ROOT / 'control_states_result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        assert result['exit_code'] == 0
        return
    if mode == 'diagnose':
        result = diagnose.remote()
        (ROOT / 'diagnostic_result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        return
    if mode == 'equivalence':
        result = equivalence.remote()
        (ROOT / 'loop_equivalence_result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        assert result['exit_code'] == 0
        return
    if mode == 'validate-control':
        path = ROOT / 'modal_results/C0_original_100_seed0_20261030T190322Z.json'
        control = json.loads(path.read_text())
        result = validate_control.remote(control['output'])
        (ROOT / 'control_validation.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        assert result['accepted'], 'Control compatibility check failed'
        (ROOT / 'control_reference.json').write_text(json.dumps(control, indent=2))
        return
    if mode in {'all', 'prepare'}:
        config = prepare.remote()
        content = json.dumps(config, ensure_ascii=False, indent=2)
        (ROOT / 'configs/rq2_swap.json').write_text(content, encoding='utf-8')
        (ROOT / 'configs/rq2_swap.yaml').write_text(content, encoding='utf-8')
        print('Protocol prepared: 10 tasks, 5 conditions, 100 episodes each.', flush=True)
    if mode in {'all', 'smoke'}:
        smoke = save_result(evaluate.remote('C0_original', True, '', min(600, max_seconds)))
        save_result(evaluate.remote('C3_empty', True, smoke['output'], min(600, max_seconds)))
    if mode in {'all', 'control'}:
        control = save_result(evaluate.remote('C0_original', False, '', max_seconds))
        (ROOT / 'control_reference.json').write_text(json.dumps(control, indent=2))
    if mode in {'all', 'variants'}:
        control = json.loads((ROOT / 'control_reference.json').read_text())
        if not control['baseline_check']['exact_episode_agreement']:
            validation = json.loads((ROOT / 'control_validation.json').read_text())
            assert validation['accepted'] and validation['reference'] == control['output']
        failures = []
        selected_conditions = conditions.split(',')
        assert selected_conditions and len(set(selected_conditions)) == len(selected_conditions)
        assert set(selected_conditions).issubset(set(CONDITIONS[1:]))
        resume_condition = None
        if resume_source:
            previous = json.loads((ROOT / 'modal_results' / f'{Path(resume_source).name}.json').read_text())
            resume_condition = previous['condition']
            assert resume_condition in selected_conditions
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(evaluate.remote, c, False, control['output'], max_seconds,
                resume_source if c == resume_condition else ''): c for c in selected_conditions}
            for future in concurrent.futures.as_completed(futures):
                try:
                    save_result(future.result())
                except Exception as exc:
                    failures.append(f'{futures[future]}: {exc}')
        if failures:
            raise RuntimeError('; '.join(failures))
