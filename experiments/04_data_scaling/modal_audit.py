"""Replay final checkpoints solely to repair reset metadata; never train or select models."""
import json
from pathlib import Path
import modal
from modal_act import image, volume, ROOT

app = modal.App('knightec-vla-stage3b-audit')
audit_image = image.add_local_file(ROOT/'audit_evaluate.py', '/opt/rq1/audit_evaluate.py', copy=True)


@app.function(image=audit_image, volumes={'/data':volume}, gpu='A10',
    cpu=(4,4), memory=(16384,16384), timeout=300, startup_timeout=120,
    max_containers=4, scaledown_window=2, retries=0, single_use_containers=True)
def audit_one(run_name):
    import hashlib
    import os
    import subprocess
    import sys
    import time
    started = time.monotonic()
    assert Path(run_name).name == run_name and run_name.startswith('formal_A10_')
    volume.reload()
    source = Path('/data/results')/run_name
    m = json.loads((source/'manifest.json').read_text())
    assert m['status'] == 'completed' and m['steps'] == 6000
    checkpoint = Path(m['checkpoint']).resolve()
    assert checkpoint.is_relative_to(source.resolve())
    checksum = hashlib.file_digest(open(checkpoint/'model.safetensors','rb'), 'sha256').hexdigest()
    assert checksum == m['checkpoint_sha256']
    output = Path('/data/audit_results')/run_name
    output.mkdir(parents=True, exist_ok=False)
    metadata = {'n_demonstrations':m['n_demonstrations'], 'train_seed':m['train_seed'],
        'training_steps':m['steps'], 'eval_seed_start':10000, 'dataset_revision':m['dataset_revision'],
        'lerobot_commit':m['lerobot_commit'], 'model':'act', 'checkpoint_sha256':checksum}
    command = [sys.executable, '/opt/rq1/audit_evaluate.py', f'--policy.path={checkpoint}',
        '--policy.device=cuda','--env.type=aloha','--env.task=AlohaTransferCube-v0',
        '--eval.batch_size=10','--eval.n_episodes=50','--seed=10000', f'--output_dir={output/"eval"}']
    run = {**metadata,'kind':'metadata_audit_replay','output':str(output),'original_run':run_name,
        'status':'running','reason':'Preserve seeded initial qpos across unseeded SAME_STEP autoresets',
        'source_sha256':hashlib.sha256(Path('/opt/rq1/audit_evaluate.py').read_bytes()).hexdigest()}
    try:
        import numpy as np
        from safetensors.numpy import load_file
        expected_stats = json.loads(Path('/opt/rq1/subset_stats.json').read_text())[str(m['n_demonstrations'])]
        stats_audit = {'n_demonstrations':m['n_demonstrations'], 'checked':[], 'subset_stats':expected_stats}
        for filename, features in [('policy_preprocessor.json',['observation.state','action']),
                                    ('policy_postprocessor.json',['action'])]:
            processor = json.loads((checkpoint/filename).read_text())
            stat_step = next(s for s in processor['steps'] if 'state_file' in s)
            tensors = load_file(str(checkpoint/stat_step['state_file']))
            for feature in features:
                for stat in ['mean','std']:
                    np.testing.assert_allclose(tensors[f'{feature}.{stat}'],
                        np.asarray(expected_stats[feature][stat], dtype=np.float32), rtol=1e-6, atol=1e-7)
                    stats_audit['checked'].append(f'{filename}:{feature}.{stat}')
            if filename == 'policy_preprocessor.json':
                for stat, values in [('mean',[.485,.456,.406]),('std',[.229,.224,.225])]:
                    np.testing.assert_allclose(tensors[f'observation.images.top.{stat}'].reshape(-1),
                                               values, rtol=1e-6, atol=1e-7)
        (output/'normalization_audit.json').write_text(json.dumps(stats_audit,indent=2))
        run['saved_normalization_verified'] = True
        with (output/'eval.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=250,
                env={**os.environ,'RQ1_EVAL_OUTPUT':str(output/'eval'),'RQ1_EVAL_METADATA':json.dumps(metadata)})
        assert result.returncode == 0, (output/'eval.log').read_text()[-3000:]
        key = lambda e:e['environment_seed']
        old = sorted([json.loads(x) for x in (source/'eval/episodes.jsonl').read_text().splitlines()], key=key)
        new = sorted([json.loads(x) for x in (output/'eval/episodes.jsonl').read_text().splitlines()], key=key)
        assert len(new) == len(old) == 50
        run['matching_outcomes'] = all(a['success']==b['success'] for a,b in zip(old,new))
        run['matching_steps'] = all(a['steps']==b['steps'] for a,b in zip(old,new))
        run['outcome_differences'] = [{'environment_seed':a['environment_seed'],
            'original_success':a['success'],'replay_success':b['success']}
            for a,b in zip(old,new) if a['success']!=b['success']]
        run['step_difference_count'] = sum(a['steps']!=b['steps'] for a,b in zip(old,new))
        run['results'] = json.loads((output/'eval/eval_info.json').read_text())
        run['status'] = 'completed'
    except Exception as exc:
        run.update(status='failed', error=str(exc))
    finally:
        run['measured_seconds'] = time.monotonic()-started
        (output/'manifest.json').write_text(json.dumps(run, indent=2))
        volume.commit()
        run['wall_seconds'] = time.monotonic()-started
    return run


@app.local_entrypoint()
def main():
    import concurrent.futures
    import time
    expected = {(n,s) for n in [5,10,20,35,50] for s in [0,1,2]}
    ledger = json.loads((ROOT/'budget_ledger.json').read_text())
    assert all(j['state'] != 'reserved' for j in ledger['jobs']), 'Wait for the formal parent to finish writing its ledger'
    path = ROOT/'audit_budget_ledger.json'
    committed = sum(j.get('settled_estimate_usd',j['reserved_usd']) for j in ledger['jobs'])
    overhead_path=ROOT/'audit_overhead_ledger.json'
    if overhead_path.exists():
        committed+=sum(e.get('settled_estimate_usd',e['reserved_usd']) for e in json.loads(overhead_path.read_text()))
    retry_path = ROOT/'retry_budget_ledger.json'
    if retry_path.exists():
        retry = json.loads(retry_path.read_text())
        if not any(j['label']==retry['label'] for j in ledger['jobs']):
            committed += retry.get('settled_estimate_usd',retry['reserved_usd'])
    entries = json.loads(path.read_text()) if path.exists() else [dict(label=f'metadata_audit_n{n}_s{s}', n=n, seed=s, gpu='A10',
        timeout_seconds=300,usd_per_second=.00039392,reserved_usd=422*.00039392,state='reserved')
        for n,s in sorted(expected)]
    assert committed+sum(e['reserved_usd'] for e in entries)<=13.5
    path.write_text(json.dumps(entries,indent=2))
    destination = ROOT/'audit_results'
    destination.mkdir(exist_ok=True)
    assert all(e['state'] in ['reserved','completed','completed_with_replay_differences'] for e in entries), 'Review failed audits first'
    started = {(e['n'],e['seed']) for e in entries if e['state']!='reserved'}
    finished = len(started)
    deadline = time.monotonic()+2100
    # Four replay GPUs can overlap with the one replacement training GPU.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        while finished < 15:
            assert time.monotonic()<deadline, 'Audit coordinator deadline exceeded'
            available = {}
            for p in sorted((ROOT/'modal_results').glob('formal_A10_*.json')):
                try: m=json.loads(p.read_text())
                except json.JSONDecodeError: continue
                assert m['status']=='completed'
                key=m['n_demonstrations'],m['train_seed']
                assert key in expected and key not in available
                available[key]=Path(m['output']).name
            for key,name in available.items():
                if key not in started and len(futures)<4:
                    started.add(key)
                    futures[pool.submit(audit_one.remote,name)]=key
            if not futures:
                time.sleep(5)
                continue
            done,_=concurrent.futures.wait(futures,timeout=5,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                key=futures.pop(future)
                result=future.result()
                name=result['original_run']
                (destination/f'{name}.json').write_text(json.dumps(result,indent=2))
                entry=next(e for e in entries if (e['n'],e['seed'])==key)
                state = 'completed_with_replay_differences' if result['status']=='completed' and not result['matching_outcomes'] else result['status']
                entry.update(state=state,run='audit_results/'+name,
                    settled_estimate_usd=min(entry['reserved_usd'],(result['wall_seconds']+122)*entry['usd_per_second']))
                path.write_text(json.dumps(entries,indent=2))
                finished+=1
                print(json.dumps({k:result.get(k) for k in ['original_run','status','wall_seconds','matching_outcomes','matching_steps','error']}),flush=True)
                assert result['status']=='completed', 'Inspect failed metadata replay'
