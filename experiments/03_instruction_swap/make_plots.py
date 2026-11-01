"""Validate paired records and reproduce language-intervention statistics and figures."""
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
LABELS = {'C0_original': 'Original', 'C1_paraphrase': 'Paraphrased',
          'C2_swap': 'Swapped task', 'C3_empty': 'Empty task text', 'C4_unrelated': 'Unrelated text'}
PAIR_FIELDS = ['initial_state_index', 'supplied_state_sha256', 'settled_state_sha256',
               'visual_model_hashes', 'cpu_rng_sha256', 'cuda_rng_sha256']


def array_hash(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def wilson(k, n):
    z = 1.959963984540054
    p = k/n
    den = 1+z*z/n
    center = (p+z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [0. if k == 0 else max(0., 100*(center-half)),
            100. if k == n else min(100., 100*(center+half))]


def exact_mcnemar(b, c):
    n = b+c
    return min(1., 2*sum(math.comb(n, i) for i in range(min(b, c)+1))/2**n) if n else 1.


def holm(p_values):
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.] * len(order)
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, (len(order)-rank)*p_values[index])
        adjusted[index] = min(1., running)
    return adjusted


def load_runs():
    config = json.loads((ROOT / 'configs/rq2_swap.json').read_text())
    digest = hashlib.sha256((ROOT / 'run_swap.py').read_bytes()).hexdigest()
    config_hash = hashlib.sha256((ROOT / 'configs/rq2_swap.json').read_bytes()).hexdigest()
    pairs, runs = {}, {}
    for condition in LABELS:
        candidates = sorted((ROOT / 'modal_results').glob(f'{condition}_100_seed0_*.json'))
        allowed = {'completed', 'baseline_mismatch'} if condition == 'C0_original' else {'completed', 'completed_aggregate'}
        good = [json.loads(p.read_text()) for p in candidates if json.loads(p.read_text())['status'] in allowed]
        if not good:
            raise RuntimeError(f'Missing completed run: {condition}')
        run = good[-1]
        assert run['runner_sha256'] == digest and run['config_sha256'] == config_hash
        assert run['model_revision'] == config['model_revision'] and run['model_repo'] == config['model_repo']
        assert run['lerobot_commit'] == config['lerobot_commit']
        if run.get('validation_wrapper_sha256'):
            assert run['validation_wrapper_sha256'] == hashlib.sha256((ROOT / 'audited_runner.py').read_bytes()).hexdigest()
        folder = ROOT / 'modal_results' / Path(run['output']).name
        records = [json.loads(line) for line in (folder / 'episodes.jsonl').read_text().splitlines()]
        assert len(records) == 100
        keyed = {(r['task_id'], r['episode']): r for r in records}
        assert len(keyed) == 100 and set(keyed) == {(t, e) for t in range(10) for e in range(10)}
        assert len(list((folder / 'videos').rglob('*.mp4'))) == 100
        expected = {t['task_id']: t['metrics']['successes'] for t in run['results']['per_task']}
        for (task_id, episode), row in keyed.items():
            assert row['success'] == expected[task_id][episode]
            assert row['condition'] == condition and row['environment_seed'] == episode
            assert row['model_revision'] == config['model_revision'] and row['lerobot_commit'] == config['lerobot_commit']
            assert row['instruction'] == config['tasks'][task_id]['instructions'][condition]
            assert row['original_instruction'] == config['tasks'][task_id]['instructions']['C0_original']
            assert row['first_formatted_prompt'].partition(', State:')[0] == 'Task: '+row['instruction'].strip().replace('_',' ').replace('\n',' ')
            assert 1 <= row['steps'] <= 300
            if row.get('state_alignment_applied'):
                assert row['state_alignment_max_abs_difference'] <= run['validation_overrides']['max_initial_state_alignment_difference']
            original_folder = ROOT / 'modal_results' / row.get('source_run', Path(run['output']).name)
            with np.load(original_folder / 'initial_images' / f'task{task_id}_episode{episode}.npz') as images:
                assert {key:array_hash(images[key]) for key in images.files} == row['initial_image_hashes']
        assert sum(r['success'] for r in records) == run['results']['overall']['n_success']
        assert run['results']['overall']['n_episodes'] == 100
        pairs[condition], runs[condition] = keyed, run
    base = pairs['C0_original']
    cache = ROOT / 'control_states' / Path(runs['C0_original']['output']).name
    cache_info = json.loads((cache / 'manifest.json').read_text())
    assert cache_info['verified_states'] == 100 and cache_info['control_output'] == runs['C0_original']['output']
    for (task_id,episode), row in base.items():
        assert array_hash(np.load(cache / f'task{task_id}_episode{episode}.npy')) == row['settled_state_sha256']
    if not runs['C0_original']['baseline_check']['exact_episode_agreement']:
        validation = json.loads((ROOT / 'control_validation.json').read_text())
        assert validation['accepted'] and validation['reference'] == runs['C0_original']['output']
        assert validation['runner_sha256'] == digest and validation['config_sha256'] == config_hash
    for condition, keyed in pairs.items():
        if condition == 'C0_original':
            continue
        assert runs[condition]['reference'] == runs['C0_original']['output']
        tolerance = runs[condition].get('validation_overrides', {}).get('image_pair_tolerance', config['image_pair_tolerance'])
        for key, row in keyed.items():
            assert row['paired_reference_checked']
            for field in PAIR_FIELDS:
                assert row[field] == base[key][field], (condition, key, field)
            for metrics in row['image_pair_metrics'].values():
                assert metrics['max_abs_difference'] <= tolerance['max_abs_channel_difference']
                assert metrics['fraction_different_channels'] <= tolerance['max_fraction_different_channels']
            assert row['first_tokens_sha256'] != base[key]['first_tokens_sha256']
            assert row['first_formatted_prompt'].partition(', State:')[2] == base[key]['first_formatted_prompt'].partition(', State:')[2]
    return config, pairs, runs


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    config, pairs, runs = load_runs()
    output = ROOT / 'results/rq2'
    output.mkdir(parents=True, exist_ok=True)
    base = pairs['C0_original']
    baseline_validation = json.loads((ROOT / 'control_validation.json').read_text()) if (ROOT / 'control_validation.json').exists() else runs['C0_original']['baseline_check']
    baseline_successes = sum(r['success'] for r in base.values())
    summaries, comparisons, task_rows, all_records = [], [], [], []
    for condition, records in pairs.items():
        k = sum(r['success'] for r in records.values())
        ci = wilson(k, 100)
        summaries.append({'condition': condition, 'n': 100, 'successes': k,
            'success_percent': k, 'ci95_low_percent': ci[0], 'ci95_high_percent': ci[1],
            'drop_vs_control_pp': baseline_successes-k, 'wall_seconds': runs[condition]['wall_seconds']})
        all_records.extend({'model': config['model_repo'], 'suite': config['suite'],
            'seed': row['run_seed'], 'score_goal': 'original_task',
            'source_run': Path(runs[condition]['output']).name, **row} for row in records.values())
        for task_id in range(10):
            subset = [records[(task_id, ep)] for ep in range(10)]
            task_rows.append({'condition': condition, 'task_id': task_id,
                'task': config['tasks'][task_id]['name'], 'n': 10,
                'successes': sum(r['success'] for r in subset),
                'instruction': subset[0]['instruction']})
        if condition != 'C0_original':
            both = sum(base[key]['success'] and row['success'] for key, row in records.items())
            b = sum(base[key]['success'] and not row['success'] for key, row in records.items())
            c = sum(not base[key]['success'] and row['success'] for key, row in records.items())
            neither = 100-both-b-c
            assert b-c == baseline_successes-k
            comparisons.append({'control': 'C0_original', 'condition': condition,
                'both_success': both, 'control_only_success': b,
                'condition_only_success': c, 'both_failure': neither,
                'paired_difference_pp': k-baseline_successes, 'mcnemar_p_exact_two_sided': exact_mcnemar(b, c)})
    for row, value in zip(comparisons, holm([r['mcnemar_p_exact_two_sided'] for r in comparisons]), strict=True):
        row['holm_p_four_comparisons'] = value
    write_csv(output / 'summary.csv', summaries)
    write_csv(output / 'paired_comparisons.csv', comparisons)
    write_csv(output / 'per_task.csv', task_rows)
    (output / 'episodes.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in all_records), encoding='utf-8')
    for condition, run in runs.items():
        (output / f'{condition}.json').write_text(json.dumps(run, indent=2))
    formal_seconds = sum(r['wall_seconds'] for r in runs.values())
    smoke_seconds = sum(json.loads(p.read_text())['wall_seconds'] for p in (ROOT / 'modal_results').glob('*_smoke_seed0_*.json'))
    segments = []
    for run in runs.values():
        if run.get('timing_segments'):
            segments.extend((s['start_unix'], s['end_unix']) for s in run['timing_segments'])
        else:
            start = datetime.datetime.strptime(Path(run['output']).name.rsplit('_', 1)[1], '%Y%m%dT%H%M%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()
            segments.append((start, start+run['wall_seconds']))
    elapsed = max(end for _,end in segments)-min(start for start,_ in segments)
    merged = []
    for start,end in sorted(segments):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1],end)
        else:
            merged.append([start,end])
    union_seconds = sum(end-start for start,end in merged)
    limits = [
        'One run seed, ten tasks and ten fixed initial states per task. Related episodes may be correlated; Wilson and McNemar statistics do not model task-level clustering.',
        'All success predicates refer to the original task. A failed original task does not prove the replacement task was completed.',
        'One fixed cyclic swap, one published paraphrase per task, one unrelated sentence. These do not cover all possible language interventions.',
        'Empty task text retains the model prompt template and proprioceptive state tokens; this is not a no-language-token ablation.',
        'Changing task text also changes token sequence lengths and state-token positions. This experiment measures task-text sensitivity; it does not fully isolate semantic understanding from formatting effects.',
        'Supplied states, settled states, visual-model parameters and initial CPU/CUDA RNG match exactly. First-image checks use each run\'s declared one- or two-level intensity tolerance in at most 0.1% of color channels. The two-level validation-only allowance was calibrated from reset-only probes; no images or actions were modified. Raw images and differences are saved.',
        'Four exact two-sided McNemar tests are reported with Holm adjustment across all four. C2 and C3 were designated primary before formal outcomes.',
        'C0 was retained even though individual outcomes can differ from stage one. A real-observation replay verified identical official/custom action tensors; compatibility validation and any baseline discrepancy are disclosed separately. No C0 reruns were selected for favorable outcomes.',
        'Cross-platform floating-point reset tails are aligned to verified C0 state snapshots only when the maximum difference is <=1e-12. Cached observation/controller sampling timing is preserved. Source snapshots exactly match the hashes originally recorded by C0. Actual alignment sizes are recorded.',
        'Interrupted C3 prefixes are retained and only missing pairs are resumed; source runs and the non-overlapping merge are recorded. No completed episode is replaced based on its outcome.',
    ]
    summary = {'model_repo': config['model_repo'], 'model_revision': config['model_revision'],
        'lerobot_commit': config['lerobot_commit'], 'seed': 0, 'n_episodes': 500,
        'conditions': summaries, 'paired_comparisons': comparisons, 'limitations': limits,
        'baseline_validation': baseline_validation,
        'all_400_intervention_pairs_verified': True,
        'pairing_audit': {'verified_control_state_vectors': 100,
            'verified_initial_image_files': 500,
            'max_applied_state_alignment_difference': max(r.get('state_alignment_max_abs_difference',0.) for r in all_records),
            'max_initial_image_channel_difference': max(m['max_abs_difference'] for r in all_records for m in (r.get('image_pair_metrics') or {}).values()),
            'max_initial_image_fraction_different_channels': max(m['fraction_different_channels'] for r in all_records for m in (r.get('image_pair_metrics') or {}).values())},
        'summed_formal_gpu_process_seconds': formal_seconds,
        'approx_formal_elapsed_seconds': elapsed,
        'approx_parallel_overlap_seconds': formal_seconds-union_seconds,
        'gaps_between_contributing_runs_seconds': elapsed-union_seconds,
        'timing_note': 'Elapsed time includes diagnostic/setup gaps. Summed process time covers runs contributing formal records, excluding zero-episode failed attempts and separate diagnostics.',
        'smoke_seconds': smoke_seconds}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    # Plot directly from the exported summary CSV, so it is independently reproducible.
    with (output / 'summary.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    y = [float(r['success_percent']) for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5.8), layout='constrained')
    x = list(range(5))
    colors = ['#64748b', '#4C78A8', '#E45756', '#F58518', '#72B7B2']
    ax.bar(x, y, color=colors, width=.62, zorder=3)
    ax.errorbar(x, y, yerr=[[float(r['success_percent'])-float(r['ci95_low_percent']) for r in rows],
        [float(r['ci95_high_percent'])-float(r['success_percent']) for r in rows]], fmt='none',
        color='#172536', capsize=5, linewidth=1.6, zorder=4)
    for i, row in enumerate(rows):
        value = float(row['success_percent'])
        ax.text(i, value/2 if value >= 20 else float(row['ci95_high_percent'])+3,
                f'{row["successes"]}/100', ha='center', va='center', fontsize=12,
                color='white' if value >= 20 else '#172536', weight='bold')
    ax.axhline(baseline_successes, linestyle='--', color='#334155', linewidth=1)
    ax.set(xticks=x, xticklabels=list(LABELS.values()), ylim=(0, 111),
           ylabel='Original-task success (%)', title='π0.5: does changing the instruction change behavior?')
    ax.grid(axis='y', alpha=.2, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    fig.supxlabel('100 paired episodes per condition • same initial states and initial policy RNG\nError bars: 95% Wilson intervals • Replacement-task completion is not measured', fontsize=9)
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(output / f'instruction_sensitivity.{extension}', dpi=200)
    plt.close(fig)
    report = ['Knightec VLA — Stage 3A: paired language interventions', '',
        f'Model: {config["model_repo"]}', '500 formal episodes; one run seed; 100 paired initial states.',
        f'C0: {baseline_successes}/100; previous stage-one baseline: 98/100. See baseline_validation in summary.json.',
        'Official/custom loop action equality was tested on identical real observations; the original C0 run was retained.', '',
        *[f'{LABELS[r["condition"]]}: {r["successes"]}/100; Wilson 95% CI {r["ci95_low_percent"]:.2f}–{r["ci95_high_percent"]:.2f}%; drop versus C0: {r["drop_vs_control_pp"]:+.1f} pp.' for r in summaries],
        '', 'Paired exact McNemar comparisons:',
        *[f'{r["condition"]}: control-only successes={r["control_only_success"]}, intervention-only successes={r["condition_only_success"]}, p={r["mcnemar_p_exact_two_sided"]:.6g}, Holm p={r["holm_p_four_comparisons"]:.6g}.' for r in comparisons],
        '', f'Elapsed time from first formal run to last completion (including diagnostic gaps): approximately {elapsed/60:.1f} minutes; sum of contributing GPU-process times: {formal_seconds/60:.1f} minutes.',
        '', 'Interpretation limits:', *['- '+line for line in limits],
        '', 'Reproduce: make plots (WSL). See README.txt and configs/rq2_swap.yaml.',
        'All 500 episode records and videos are saved locally.']
    (output / 'run-report.txt').write_text('\n'.join(report)+'\n')
    print(json.dumps({'conditions': summaries, 'paired_comparisons': comparisons,
                      'formal_elapsed_minutes': elapsed/60}, indent=2))


if __name__ == '__main__':
    main()
