"""Summarize one official evaluation output, preserving task-level outcomes."""
import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('run_name')
args = parser.parse_args()
root = Path(__file__).parent / 'modal_results'
run = json.loads((root / f'{args.run_name}.json').read_text())
info = run['results']
assert run['status'] == 'completed'
if run['mode'] == 'goal':
    assert info['overall']['n_episodes'] == 100
    assert len(info['per_task']) == 10
    assert {task['task_id'] for task in info['per_task']} == set(range(10))
rows = []
episodes = []
for task in info['per_task']:
    outcomes = task['metrics']['successes']
    assert len(outcomes) == task['n_episodes']
    assert sum(outcomes) == task['n_success']
    rows.append({'suite': task['task_group'], 'task_id': task['task_id'],
                 'n': task['n_episodes'], 'successes': task['n_success'],
                 'success_percent': task['pc_success'],
                 'ci95_low_percent': task['pc_success_ci95'][0],
                 'ci95_high_percent': task['pc_success_ci95'][1]})
    for index, success in enumerate(outcomes):
        episodes.append({'model': run['model_repo'], 'suite': task['task_group'],
                         'task_id': task['task_id'], 'episode': index,
                         'run_seed': run['seed'], 'success': success,
                         'model_revision': run['model_revision'],
                         'lerobot_commit': run['lerobot_commit']})
assert sum(row['successes'] for row in rows) == info['overall']['n_success']
destination = root / args.run_name
destination.mkdir(exist_ok=True)
with (destination / 'task_summary.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
with (destination / 'episodes.jsonl').open('w') as stream:
    for episode in episodes:
        stream.write(json.dumps(episode) + '\n')
summary = {'model': run['model_repo'], 'model_revision': run['model_revision'],
           'mode': run['mode'], 'seed': run['seed'], 'gpu': run['gpu'],
           'overall': info['overall'], 'per_task': rows,
           'wall_seconds': run['wall_seconds'],
           'official_goal_reference_percent': 98.0,
           'goal_difference_percentage_points': info['overall']['pc_success'] - 98.0,
           'limitation': 'One evaluation seed; Goal suite only. Wilson intervals treat episode outcomes as Bernoulli observations.'}
(destination / 'summary.json').write_text(json.dumps(summary, indent=2))
overall = info['overall']
report = [
    'Knightec VLA — Stage 1: LIBERO-Goal baseline',
    '',
    f"Model: {run['model_repo']}",
    f"Success: {overall['n_success']}/{overall['n_episodes']} ({overall['pc_success']:.1f}%)",
    f"95% Wilson interval: {overall['pc_success_ci95'][0]:.2f}% to {overall['pc_success_ci95'][1]:.2f}%",
    f"Official Goal reference: 98.0%; difference: {overall['pc_success'] - 98.0:+.1f} percentage points",
    f"Evaluation time: {overall['eval_s']:.1f} seconds; {overall['eval_ep_s']:.2f} seconds/episode",
    f"Runtime including model loading: {run['wall_seconds']:.1f} seconds",
    f"GPU: {run['gpu']}; seed: {run['seed']}",
    f"LeRobot commit: {run['lerobot_commit']}",
    f"Model revision: {run['model_revision']}",
    '',
    'Scope: 10 Goal tasks, 10 episodes/task, one evaluation seed. Other suites and seeds were not evaluated.',
    'Settings: batch size 1, hard reset, fixed initial states, relative control, 10 action steps, BF16, compilation disabled.',
    'Reference: https://huggingface.co/docs/lerobot/en/libero',
    '',
    'Per-task outcomes:',
    *[f"Task {row['task_id']}: {row['successes']}/{row['n']}" for row in rows],
]
(destination / 'run-report.txt').write_text('\n'.join(report) + '\n')
print(json.dumps(summary, indent=2))
