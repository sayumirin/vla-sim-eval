"""Merge an immutable completed prefix with its non-overlapping resumed episodes."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
from make_plots import wilson

ROOT = Path(__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument('resume_run')
args = parser.parse_args()
assert Path(args.resume_run).name == args.resume_run
resumed = json.loads((ROOT / 'modal_results' / f'{args.resume_run}.json').read_text())
assert resumed['status'] == 'completed' and resumed['resume_source']
prefix_name = Path(resumed['resume_source']).name
prefix = json.loads((ROOT / 'modal_results' / f'{prefix_name}.json').read_text())
assert prefix['condition'] == resumed['condition']
for key in ['runner_sha256', 'config_sha256', 'reference', 'model_revision', 'lerobot_commit']:
    assert prefix[key] == resumed[key], key
stamp = Path(resumed['output']).name.rsplit('_', 1)[1]
destination = ROOT / 'modal_results' / f'{resumed["condition"]}_100_seed0_{stamp}'
assert not (destination / 'eval.log').exists(), 'Do not overwrite a real run'
destination.mkdir(exist_ok=True)
records = {}
timing_segments = []
source_counts = {}
for run in [prefix, resumed]:
    name = Path(run['output']).name
    source = ROOT / 'modal_results' / name
    rows = [json.loads(line) for line in (source / 'episodes.jsonl').read_text().splitlines()]
    source_counts[name] = len(rows)
    start = datetime.datetime.strptime(name.rsplit('_',1)[1], '%Y%m%dT%H%M%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()
    timing_segments.append({'source_run': name, 'start_unix': start,
                           'end_unix': start+run['wall_seconds'], 'wall_seconds': run['wall_seconds']})
    offsets = {int(k):int(v) for k,v in run.get('resume_offsets', {}).items()}
    if run['status'] == 'completed':
        expected = {(t['task_id'], i+offsets.get(t['task_id'],0)): bool(v)
                    for t in run['results']['per_task'] for i,v in enumerate(t['metrics']['successes'])}
        assert {(r['task_id'],r['episode']): r['success'] for r in rows} == expected
        assert len(rows) == run['results']['overall']['n_episodes']
    for row in rows:
        key = (row['task_id'], row['episode'])
        assert key not in records, f'Duplicate pair {key}'
        index = row['episode']-offsets.get(row['task_id'], 0)
        video = source / 'videos' / f'libero_goal_{row["task_id"]}' / f'eval_episode_{index}.mp4'
        assert video.is_file() and video.stat().st_size > 0
        target = destination / 'videos' / f'libero_goal_{row["task_id"]}' / f'eval_episode_{row["episode"]}.mp4'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video, target)
        records[key] = {**row, 'source_run': name,
                        'source_video': str(video.relative_to(ROOT)),
                        'video_relpath': str(target.relative_to(destination))}
assert set(records) == {(t,e) for t in range(10) for e in range(10)}
assert sum(source_counts.values()) == 100
tasks = []
for task_id in range(10):
    successes = [records[(task_id,e)]['success'] for e in range(10)]
    k = sum(successes)
    tasks.append({'task_group': 'libero_goal', 'task_id': task_id,
        'metrics': {'successes': successes}, 'n_episodes': 10, 'n_success': k,
        'pc_success': k*10., 'pc_success_ci95': wilson(k, 10)})
k = sum(r['success'] for r in records.values())
info = {'per_task': tasks, 'overall': {'n_episodes': 100, 'n_success': k,
        'pc_success': float(k), 'pc_success_ci95': wilson(k,100)},
        'aggregation_note': 'Only episode success outcomes are aggregated. Rewards are not reconstructed.'}
run = {**resumed, 'output': str(destination), 'status': 'completed_aggregate',
       'aggregation_only': True, 'storage': 'local_aggregate',
       'command': None, 'gpu': None, 'cloud': None, 'region': None,
       'resume_source': None, 'resume_offsets': {}, 'preflight': None,
       'hardware_variants': [{'source_run': Path(r['output']).name, 'gpu': r['gpu'],
                              'cloud': r.get('cloud'), 'region': r.get('region')} for r in [prefix,resumed]],
       'source_runs': [prefix_name, args.resume_run], 'source_counts': source_counts,
       'timing_segments': timing_segments, 'results': info,
       'expected_episodes_this_run': 100,
       'wall_seconds': sum(s['wall_seconds'] for s in timing_segments),
       'aggregation_rule': 'Retain every completed prefix episode and append only previously unattempted pairs; no outcome-based selection and no duplicate pairs.'}
ordered = [records[key] for key in sorted(records)]
(destination / 'episodes.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in ordered), encoding='utf-8')
(destination / 'eval_info.json').write_text(json.dumps(info, indent=2))
(destination / 'manifest.json').write_text(json.dumps(run, indent=2))
(ROOT / 'modal_results' / f'{destination.name}.json').write_text(json.dumps(run, indent=2))
print(json.dumps({'aggregate': destination.name, 'source_counts': source_counts,
                  'successes': k, 'episodes': 100, 'videos': 100}, indent=2))
