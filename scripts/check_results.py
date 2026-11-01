"""Recompute every reported number from the per-episode records and compare with the committed summaries.

Needs only numpy. Run from the repository root:  python scripts/check_results.py
"""
import json
import math
import sys

import load
from stats import exact_mcnemar, holm, two_way_bootstrap, wilson

failures = []


def check(label, recomputed, reported, tol=1e-9):
    ok = math.isclose(float(recomputed), float(reported), rel_tol=tol, abs_tol=tol)
    if not ok:
        failures.append(f'{label}: recomputed {recomputed} != reported {reported}')
    return ok


def baseline():
    rows = load.baseline()
    k = sum(r['success'] for r in rows)
    reported = json.loads((load.EXPERIMENTS / '01_baseline/results/summary.json').read_text())['overall']
    low, high = wilson(k, 100)
    check('baseline successes', k, reported['n_success'])
    check('baseline CI low', low, reported['pc_success_ci95'][0])
    check('baseline CI high', high, reported['pc_success_ci95'][1])
    for row in load.read_csv(load.EXPERIMENTS / '01_baseline/results/task_summary.csv'):
        task = int(row['task_id'])
        check(f'baseline task {task}', sum(r['success'] for r in rows if r['task_id'] == task), row['successes'])
    print(f'1 Baseline           {k:>3}/100  CI {low:.1f}-{high:.1f}%')


def plus():
    grouped = load.plus()
    for row in load.read_csv(load.EXPERIMENTS / '02_libero_plus/results/summary.csv'):
        outcomes = grouped[row['category']]
        k = sum(outcomes)
        low, high = wilson(k, len(outcomes))
        check(f'plus {row["category"]} successes', k, row['successes'])
        check(f'plus {row["category"]} CI low', low, row['ci95_low_percent'])
        check(f'plus {row["category"]} CI high', high, row['ci95_high_percent'])
        print(f'2 Plus {row["category"]:<13} {k:>3}/100  CI {low:.1f}-{high:.1f}%')


def instruction():
    pairs = load.instruction_pairs()
    folder = load.EXPERIMENTS / '03_instruction_swap/results'
    for row in load.read_csv(folder / 'summary.csv'):
        k = sum(pairs[row['condition']].values())
        low, high = wilson(k, 100)
        check(f'{row["condition"]} successes', k, row['successes'])
        check(f'{row["condition"]} CI low', low, row['ci95_low_percent'])
        check(f'{row["condition"]} CI high', high, row['ci95_high_percent'])
    control = pairs['C0_original']
    print(f'3 C0_original       {sum(control.values()):>3}/100  (paired control)')
    reported = load.read_csv(folder / 'paired_comparisons.csv')
    p_values = []
    for row in reported:
        condition = pairs[row['condition']]
        b = sum(control[key] and not condition[key] for key in control)
        c = sum(not control[key] and condition[key] for key in control)
        check(f'{row["condition"]} control-only', b, row['control_only_success'])
        check(f'{row["condition"]} condition-only', c, row['condition_only_success'])
        p_values.append(exact_mcnemar(b, c))
        check(f'{row["condition"]} McNemar p', p_values[-1], row['mcnemar_p_exact_two_sided'])
    for row, adjusted in zip(reported, holm(p_values), strict=True):
        check(f'{row["condition"]} Holm p', adjusted, row['holm_p_four_comparisons'])
        k = sum(pairs[row['condition']].values())
        print(f'3 {row["condition"]:<17} {k:>3}/100  McNemar vs C0 p={float(row["mcnemar_p_exact_two_sided"]):.2g}')


def scaling():
    cube = load.scaling_cube()
    means = cube.mean(axis=(1, 2))
    low, high = two_way_bootstrap(cube)
    rows = load.read_csv(load.EXPERIMENTS / '04_data_scaling/results/summary.csv')
    for i, row in enumerate(rows):
        n = int(row['n_demonstrations'])
        check(f'N={n} mean', means[i], row['mean_success_rate'])
        check(f'N={n} bootstrap low', low[i], row['bootstrap_low'])
        check(f'N={n} bootstrap high', high[i], row['bootstrap_high'])
        for j, seed in enumerate(load.TRAINING_SEEDS):
            check(f'N={n} seed {seed}', cube[i, j].mean(), row[f'seed_{seed}_rate'])
        print(f'4 ACT N={n:<2} demos     {means[i]:.1%}  bootstrap {low[i]:.1%}-{high[i]:.1%}')
    for row in load.read_csv(load.EXPERIMENTS / '04_data_scaling/results/per_model.csv'):
        i = load.DEMONSTRATIONS.index(int(row['n_demonstrations']))
        j = load.TRAINING_SEEDS.index(int(row['train_seed']))
        k = int(cube[i, j].sum())
        check(f'model N={row["n_demonstrations"]} seed {row["train_seed"]}', k, row['successes'])
        check(f'model N={row["n_demonstrations"]} seed {row["train_seed"]} Wilson low', wilson(k, 50)[0] / 100, row['wilson_low'])


if __name__ == '__main__':
    baseline()
    plus()
    instruction()
    scaling()
    if failures:
        print('\nMISMATCHES:', *failures, sep='\n  ')
        sys.exit(1)
    print('\nAll reported numbers match the per-episode records.')
