"""Validate the three completed runs and regenerate all stage-two summaries."""
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
LABELS = {'light': 'Lighting', 'camera': 'Camera viewpoints', 'layout': 'Object layout'}


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    den = 1 + z*z/n
    mid = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return [100*(mid-half), 100*(mid+half)]


def main():
    config = json.loads((ROOT / 'configs/plus_tasks.json').read_text())
    baseline = json.loads((ROOT.parent / 'stage1/modal_results/goal_seed0_20261028T233602Z/summary.json').read_text())
    baseline_manifest = json.loads((ROOT.parent / 'stage1/modal_results/goal_seed0_20261028T233602Z/manifest.json').read_text())
    baseline['lerobot_commit'] = baseline_manifest['lerobot_commit']
    assert baseline['model_revision'] == baseline_manifest['model_revision']
    records, summaries, per_base, runs = [], [], [], {}
    for category in LABELS:
        candidates = sorted((ROOT / 'modal_results').glob(f'{category}_100_seed0_*.json'))
        completed = [json.loads(p.read_text()) for p in candidates if json.loads(p.read_text())['status'] == 'completed']
        if not completed:
            raise RuntimeError(f'Missing completed 100-variant run: {category}')
        run = completed[-1]
        runs[category] = run
        info = run['results']
        assert run['model_revision'] == baseline['model_revision']
        assert run['lerobot_commit'] == baseline['lerobot_commit']
        assert run['libero_plus_commit'] == config['libero_plus_commit']
        assert run['runner_sha256'] == hashlib.sha256((ROOT / 'run_plus_eval.py').read_bytes()).hexdigest()
        assert run['selection_sha256'] == hashlib.sha256((ROOT / 'configs/plus_tasks.json').read_bytes()).hexdigest()
        assert info['overall']['n_episodes'] == 100 and len(info['per_task']) == 100
        selected = {r['task_id']: r for r in config['categories'][category]['tasks']}
        assert {t['task_id'] for t in info['per_task']} == set(selected)
        folder = ROOT / 'modal_results' / Path(run['output']).name
        assert len(list((folder / 'videos').rglob('*.mp4'))) == 100
        progress = {p['task_id']: p for p in map(json.loads, (folder / 'progress.jsonl').read_text().splitlines())}
        assert set(progress) == set(selected)
        for t in info['per_task']:
            assert t['n_episodes'] == 1 and len(t['metrics']['successes']) == 1
            success = bool(t['metrics']['successes'][0])
            assert len(progress[t['task_id']]['steps']) == 1
            assert 1 <= progress[t['task_id']]['steps'][0] <= 300
            assert success == bool(progress[t['task_id']]['metrics']['successes'][0])
            assert t['n_success'] == int(success)
            r = selected[t['task_id']]
            records.append({'category': category, 'task_id': t['task_id'],
                'classification_id': r['id'], 'task': r['name'], 'base_task': r['base_task'],
                'difficulty_level': r['difficulty_level'], 'prompt': r['prompt'],
                'episode': 0, 'seed': 0, 'steps': progress[t['task_id']]['steps'][0],
                'success': success, 'model_revision': run['model_revision'],
                'lerobot_commit': run['lerobot_commit'], 'libero_plus_commit': run['libero_plus_commit']})
        category_records = [r for r in records if r['category'] == category]
        k = sum(r['success'] for r in category_records)
        assert k == info['overall']['n_success']
        base_results = []
        for base in sorted({r['base_task'] for r in category_records}):
            subset = [r for r in category_records if r['base_task'] == base]
            successes = sum(r['success'] for r in subset)
            base_results.append(successes / len(subset) * 100)
            per_base.append({'category': category, 'base_task': base, 'n': len(subset),
                             'successes': successes, 'success_percent': base_results[-1]})
        ci = wilson(k, 100)
        summaries.append({'category': category, 'n': 100, 'successes': k, 'success_percent': float(k),
            'ci95_low_percent': ci[0], 'ci95_high_percent': ci[1],
            'drop_vs_baseline_pp': baseline['overall']['pc_success'] - k,
            'macro_average_over_10_base_tasks_percent': sum(base_results)/len(base_results),
            'wall_seconds': run['wall_seconds'], 'run': Path(run['output']).name})
    assert len(records) == 300
    output = ROOT / 'results/plus'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    for filename, rows in [('summary.csv', summaries), ('per_base_task.csv', per_base)]:
        with (output / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for category, run in runs.items():
        (output / f'{category}.json').write_text(json.dumps(run, indent=2))
    limitations = [
        'One episode per sampled variant, evaluation and sampling seed 0; not the full LIBERO-plus benchmark.',
        '100 variants sampled uniformly without replacement per dimension. Base-task and difficulty mixtures differ; see per_base_task.csv.',
        'Wilson intervals are descriptive Bernoulli intervals; related variants may be correlated, and intervals do not include seed or sampling uncertainty.',
        'Stage-one 98/100 baseline has 10 initial states per base task; plus uses one initial state per variant, with custom layout states. This is not a paired causal comparison.',
        'The isolated plus environment requires robosuite 1.4.1, while baseline used 1.4.0; this dependency difference is another comparison limitation.',
        'Visual-perturbation prompts were normalized to original base-task prompts, removing filename suffixes. This protocol differs from unmodified LeRobot plus CLI.',
        'Object layout combines object relocation and added distractors; it does not measure distractors alone.',
        'Some upstream lighting difficulty labels are null; they are retained as unknown, without inferred labels.',
    ]
    summary = {'baseline': baseline['overall'], 'categories': summaries, 'limitations': limitations,
               'model_revision': baseline['model_revision'], 'lerobot_commit': baseline['lerobot_commit'],
               'libero_plus_commit': config['libero_plus_commit']}
    full_seconds = sum(run['wall_seconds'] for run in runs.values())
    starts = {k: datetime.datetime.strptime(Path(run['output']).name.rsplit('_', 1)[1], '%Y%m%dT%H%M%SZ').replace(tzinfo=datetime.timezone.utc).timestamp() for k, run in runs.items()}
    formal_elapsed = max(starts[k]+run['wall_seconds'] for k, run in runs.items()) - min(starts.values())
    smoke_seconds = sum(json.loads(p.read_text())['wall_seconds'] for p in (ROOT / 'modal_results').glob('*_smoke_seed0_*.json'))
    dollars_per_second = .000583 + 4*.0000131 + 32*.00000222
    summary['runtime_and_resource_estimate'] = {
        'summed_formal_process_seconds_including_model_load': full_seconds,
        'approx_formal_elapsed_wall_seconds': formal_elapsed,
        'approx_overlap_seconds_saved_by_parallel_execution': full_seconds-formal_elapsed,
        'smoke_run_seconds_including_model_load': smoke_seconds,
        'formal_run_usd_estimate': full_seconds*dollars_per_second,
        'formal_plus_smoke_usd_estimate': (full_seconds+smoke_seconds)*dollars_per_second,
        'assumed_resources': 'A100 40GB, 4 physical CPU cores, 32 GiB RAM',
        'price_source': 'https://modal.com/pricing', 'price_checked_date': '2026-10-29',
        'limitation': 'Runtime-based resource estimate, not a verified invoice. Excludes CPU preparation/image builds, container startup outside the recorded timer, idle time, storage and transfer; credits not deducted.'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    fig, ax = plt.subplots(figsize=(9, 5.5), layout='constrained')
    x = list(range(3))
    y = [s['success_percent'] for s in summaries]
    ax.bar(x, y, width=.58, color=['#4C78A8', '#F58518', '#54A24B'], zorder=3)
    ax.errorbar(x, y, yerr=[[s['success_percent']-s['ci95_low_percent'] for s in summaries],
        [s['ci95_high_percent']-s['success_percent'] for s in summaries]], fmt='none',
        color='#172536', capsize=6, linewidth=1.8, zorder=4)
    b = baseline['overall']
    ax.axhspan(*b['pc_success_ci95'], color='#64748b', alpha=.12, label='Baseline 95% Wilson interval')
    ax.axhline(b['pc_success'], color='#334155', linestyle='--', linewidth=1.6, label=f'Unperturbed Goal baseline: {b["pc_success"]:.0f}%')
    for i, s in enumerate(summaries):
        ax.text(i, max(s['success_percent']/2, 8), f'{s["successes"]}/100', ha='center', va='center',
                color='white', weight='bold', fontsize=13)
    ax.set(xticks=x, xticklabels=list(LABELS.values()), ylim=(0, 106), ylabel='Success rate (%)',
           title='π0.5 on LIBERO-Goal: visual robustness')
    ax.grid(axis='y', alpha=.2, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='lower left', framealpha=.95, fontsize=9)
    fig.supxlabel('100 distinct variants per condition • one episode each • seed 0\nError bars: 95% Wilson intervals; base-task mixtures differ.', fontsize=9)
    fig.savefig(output / 'robustness.png', dpi=200)
    fig.savefig(output / 'robustness.pdf')
    plt.close(fig)
    report = ['Knightec VLA — Stage 2: visual robustness', '',
        'Model: lerobot/pi05_libero_finetuned_v044', 'Baseline: 98/100 (98.0%)', '',
        *[f'{LABELS[s["category"]]}: {s["successes"]}/100; 95% Wilson CI {s["ci95_low_percent"]:.2f}–{s["ci95_high_percent"]:.2f}%; drop {s["drop_vs_baseline_pp"]:.1f} percentage points' for s in summaries],
        '', f'Summed formal GPU-process runtime including model loading: {full_seconds/60:.1f} minutes.',
        f'Approximate elapsed time from first formal run to last completion: {formal_elapsed/60:.1f} minutes; overlapping the last two groups saved about {(full_seconds-formal_elapsed)/60:.1f} minutes.',
        f'Formal plus smoke resource estimate: ${(full_seconds+smoke_seconds)*dollars_per_second:.2f}; not an invoice, excludes CPU build/preparation and unrecorded startup/storage/transfer.',
        '', 'Interpretation limits:', *['- '+s for s in limitations], '',
        'Reproduce summaries and plots: make plots (in stage2, WSL), or python make_plots.py.',
        'Raw results, 300 episode records, per-base-task counts and videos are preserved.']
    (output / 'run-report.txt').write_text('\n'.join(report)+'\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
