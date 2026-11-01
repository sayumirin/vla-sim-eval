"""Validate the frozen experiment and regenerate its statistics and figures offline."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'
DESIGN = json.loads((ROOT/'configs/final_design.json').read_text())
PLAN = json.loads((ROOT/'configs/rq1_plan.json').read_text())


def read(path):
    return json.loads(path.read_text())


def wilson(k, n):
    z = 1.959963984540054
    p = k/n
    denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return max(0., center-radius), min(1., center+radius)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    sizes, seeds = DESIGN['sizes'], DESIGN['training_seeds']
    expected = {(n, s) for n in sizes for s in seeds}
    runs = {}
    for path in sorted((ROOT/'modal_results').glob('formal_A10_*.json')):
        manifest = read(path)
        key = manifest['n_demonstrations'], manifest['train_seed']
        assert key not in runs, f'Duplicate formal run: {key}'
        assert manifest['status'] == 'completed', path
        runs[key] = (manifest, path.with_suffix(''))
    assert set(runs) == expected, f'Missing runs: {expected-set(runs)}'
    assert len({m['checkpoint_sha256'] for m, _ in runs.values()}) == 15
    OUT.mkdir(exist_ok=True)
    records, per_run, files, replay_comparisons = [], [], [], []
    cube = np.zeros((len(sizes), len(seeds), 50), dtype=float)
    reference_poses = None
    source_hashes = None
    for i, n in enumerate(sizes):
        for j, seed in enumerate(seeds):
            m, folder = runs[n, seed]
            assert m['steps'] == DESIGN['steps'] and m.get('start_step',0) == 0
            assert not m.get('resume_from') and not any(arg.startswith('--resume=') for arg in m['train_command'])
            assert m['dataset_revision'] == PLAN['dataset_revision']
            assert m['selected_episodes'] == PLAN['subsets'][str(n)]
            checkpoint_relative=m.get('checkpoint_relative',f'train/checkpoints/{m["steps"]:06d}/pretrained_model')
            saved_cfg = read(folder/checkpoint_relative/'train_config.json')
            assert (saved_cfg['steps'], saved_cfg['batch_size'], saved_cfg['seed']) == (6000, 8, seed)
            assert saved_cfg['dataset']['episodes'] == m['selected_episodes']
            assert saved_cfg['scheduler'] is None and saved_cfg['policy']['optimizer_lr'] == 1e-5
            assert saved_cfg['policy']['pretrained_path'] is None
            assert saved_cfg['env']['task'] == 'AlohaTransferCube-v0'
            audit = read(folder/'subset_audit.json')
            assert audit['selected_episodes'] == m['selected_episodes']
            assert audit['frames'] == audit['normalization_frames'] == n*400
            progress = [json.loads(x) for x in (folder/'training_progress.jsonl').read_text().splitlines()]
            assert progress[-1]['step'] == DESIGN['steps']
            assert all(np.isfinite(x['loss']) for x in progress)
            if source_hashes is None:
                source_hashes = m['source_hashes']
            assert source_hashes == m['source_hashes']
            # An explicitly recorded audit replay may replace faulty pose metadata;
            # outcomes are compared below, but primary scores always come from the first evaluation.
            evaluation = folder/'eval'
            replay = ROOT/'audit_results'/folder.name
            if replay.exists():
                evaluation = replay/'eval'
                replay_manifest = read(replay/'manifest.json')
                assert replay_manifest['status'] == 'completed' or replay_manifest.get('error') == 'Replay differs from original; inspect before summarizing'
                assert replay_manifest['saved_normalization_verified'] is True
                assert replay_manifest['checkpoint_sha256'] == m['checkpoint_sha256']
            ep_path = evaluation/'episodes.jsonl'
            episodes = sorted([json.loads(x) for x in ep_path.read_text().splitlines()], key=lambda x:x['environment_seed'])
            assert len(episodes) == 50
            assert [e['environment_seed'] for e in episodes] == list(range(10000, 10050))
            for e in episodes:
                assert (e['n_demonstrations'], e['train_seed'], e['training_steps']) == (n, seed, 6000)
                assert e['checkpoint_sha256'] == m['checkpoint_sha256']
                assert isinstance(e['success'], bool) and 1 <= e['steps'] <= 400
            original = [json.loads(x) for x in (folder/'eval/episodes.jsonl').read_text().splitlines()]
            original = sorted(original, key=lambda x:x['environment_seed'])
            assert len(original)==50 and [e['environment_seed'] for e in original]==list(range(10000,10050))
            for e in original:
                assert (e['n_demonstrations'],e['train_seed'],e['training_steps'])==(n,seed,6000)
                assert e['checkpoint_sha256']==m['checkpoint_sha256']
                assert isinstance(e['success'],bool) and 1<=e['steps']<=400
            successes = np.array([e['success'] for e in original], dtype=float)
            differences=[dict(environment_seed=a['environment_seed'],original_success=a['success'],replay_success=b['success'])
                         for a,b in zip(original,episodes) if a['success']!=b['success']]
            replay_comparisons.append(dict(n_demonstrations=n,train_seed=seed,outcome_differences=differences,
                original_successes=int(successes.sum()),replay_successes=sum(e['success'] for e in episodes),
                step_difference_count=sum(a['steps']!=b['steps'] for a,b in zip(original,episodes))))
            k = int(successes.sum())
            assert m['results']['overall']['n_success']==k and m['results']['overall']['n_episodes']==50
            info=read(evaluation/'eval_info.json')['overall']
            assert info['n_success']==sum(e['success'] for e in episodes) and info['n_episodes']==50
            poses = np.array([e['initial_qpos'] for e in episodes])
            assert poses.shape[0] == 50 and np.isfinite(poses).all()
            if reference_poses is None:
                reference_poses = poses
            np.testing.assert_allclose(poses, reference_poses, rtol=0, atol=1e-8)
            videos = list(evaluation.rglob('*.mp4'))
            assert len(videos) == 2 and all(v.stat().st_size > 1000 for v in videos)
            lo, hi = wilson(k, 50)
            per_run.append(dict(n_demonstrations=n, train_seed=seed, successes=k, episodes=50,
                success_rate=k/50, wilson_low=lo, wilson_high=hi, training_steps=6000,
                checkpoint_sha256=m['checkpoint_sha256'], run=folder.name))
            cube[i, j] = successes
            for primary, replay_row in zip(original,episodes):
                records.append(dict(primary, run=folder.name,
                    original_logged_reset_qpos=primary['initial_qpos'],
                    initial_qpos=replay_row['initial_qpos'],
                    initial_qpos_source='reconstructed from same-seed audit replay; original reset log was overwritten',
                    replay_success=replay_row['success'], replay_steps=replay_row['steps']))
            for input_path in [ep_path,folder/'eval/episodes.jsonl']:
                files.append(dict(path=str(input_path.relative_to(ROOT)), sha256=hashlib.sha256(input_path.read_bytes()).hexdigest()))
    # Resample the same training-seed indices and shared test cases across all N.
    rng = np.random.default_rng(20261030)
    boot = np.empty((10000, len(sizes)))
    for b in range(len(boot)):
        sampled_seeds = rng.integers(0, len(seeds), len(seeds))
        sampled_cases = rng.integers(0, 50, 50)
        boot[b] = cube[:, sampled_seeds, :][:, :, sampled_cases].mean(axis=(1, 2))
    ci = np.quantile(boot, [.025, .975], axis=0)
    means = cube.mean(axis=(1, 2))
    summary = [dict(n_demonstrations=n, mean_success_rate=means[i],
        bootstrap_low=ci[0, i], bootstrap_high=ci[1, i],
        seed_std=cube[i].mean(axis=1).std(ddof=1),
        seed_0_rate=cube[i, 0].mean(), seed_1_rate=cube[i, 1].mean(), seed_2_rate=cube[i, 2].mean(),
        successes=int(cube[i].sum()), episodes=150) for i, n in enumerate(sizes)]
    write_csv(OUT/'summary.csv', summary)
    write_csv(OUT/'per_model.csv', per_run)
    write_csv(OUT/'replay_comparison.csv',[dict(n_demonstrations=r['n_demonstrations'],train_seed=r['train_seed'],
        original_successes=r['original_successes'],replay_successes=r['replay_successes'],
        outcome_difference_count=len(r['outcome_differences']),step_difference_count=r['step_difference_count'])
        for r in replay_comparisons])
    (OUT/'episodes.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in records))
    (OUT/'audit.json').write_text(json.dumps(dict(models=15, episodes=750,
        unique_evaluation_cases=50, matched_replay_initial_qpos=True, subset_checks=True,
        primary_outcomes='first formal evaluation, never replaced by replay outcomes',
        initial_qpos_provenance='reconstructed from explicit-seed replays; original field overwritten by autoreset',
        replay_comparisons=replay_comparisons,
        shared_training_source_hashes=source_hashes, input_files=files), indent=2))
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    ax = axes[0]
    ax.fill_between(sizes, ci[0]*100, ci[1]*100, color='#2463a6', alpha=.16, label='95% two-way bootstrap')
    ax.plot(sizes, means*100, '-o', color='#2463a6', lw=2, label='Mean of 3 training seeds')
    for n, rate in zip(sizes, means):
        ax.annotate(f'{rate:.1%}', (n, rate*100), xytext=(0, 11), textcoords='offset points', ha='center')
    ax.set_title('ACT data scaling at fixed training budget')
    ax.legend(loc='lower right', fontsize=9)
    ax = axes[1]
    for j, seed in enumerate(seeds):
        rows = [r for r in per_run if r['train_seed'] == seed]
        p = np.array([r['success_rate'] for r in rows])*100
        low = np.array([r['wilson_low'] for r in rows])*100
        high = np.array([r['wilson_high'] for r in rows])*100
        x = np.array(sizes)*np.exp((j-1)*.045)
        ax.errorbar(x, p, yerr=[p-low, high-p], fmt='o', capsize=3, alpha=.8, label=f'Training seed {seed}')
    ax.set_title('Individual policies: Wilson 95% intervals')
    ax.legend(loc='lower right', fontsize=9)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xticks(sizes, labels=[str(n) for n in sizes])
        ax.set_ylim(-3, 108)
        ax.set_xlabel('Training demonstrations (nested subsets)')
        ax.set_ylabel('Success rate (%)')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('AlohaTransferCube-v0 | 6,000 updates per model | 50 shared test cases per model', fontsize=12)
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(OUT/f'data_scaling.{extension}', dpi=200)
    plt.close(fig)
    (OUT/'statistics.json').write_text(json.dumps({'bootstrap_seed':20261030,
        'bootstrap_replicates':10000,'summary':summary,
        'limitations':read(ROOT/'analysis_plan.json')['limitations']}, indent=2))
    ledger = read(ROOT/'budget_ledger.json')
    assert all(j['state'] != 'reserved' for j in ledger['jobs'])
    estimated = sum(j.get('settled_estimate_usd', j['reserved_usd']) for j in ledger['jobs'])
    formal_cost = sum(j.get('settled_estimate_usd', j['reserved_usd']) for j in ledger['jobs'] if j['label'].startswith(('formal_', 'infrastructure_retry_')))
    audit_cost = sum(j.get('settled_estimate_usd', j['reserved_usd']) for j in ledger['jobs'] if j['label'].startswith('metadata_audit_'))
    assert estimated <= 13.5
    budget = dict(user_credit_usd=15, conservative_resource_estimate_usd=estimated,
        formal_estimate_usd=formal_cost, audit_estimate_usd=audit_cost,
        note='Resource estimate including startup allowances and retained cancelled reservations; not a billing statement. CPU build, storage and transfer use separate headroom.')
    (OUT/'budget_summary.json').write_text(json.dumps(budget, indent=2))
    difference_count=sum(len(r['outcome_differences']) for r in replay_comparisons)
    changed_steps=sum(r['step_difference_count'] for r in replay_comparisons)
    lines = [
        'Experiment 4: ACT data-scaling curve', '',
        '15 models, each trained from scratch for 6,000 updates and evaluated on the same 50 test cases: 750 formal outcomes.',
        'Model: ACT (ImageNet ResNet18 initialization). Task: AlohaTransferCube-v0. At most 5 A10 GPUs in parallel.', '',
        'demos | seed 0 | seed 1 | seed 2 | mean success | descriptive bootstrap 95% interval',
    ]
    for row in summary:
        lines.append(f"{row['n_demonstrations']:>5} | {row['seed_0_rate']:6.1%} | {row['seed_1_rate']:6.1%} | {row['seed_2_rate']:6.1%} | {row['mean_success_rate']:12.1%} | {row['bootstrap_low']:.1%}–{row['bootstrap_high']:.1%}")
    lines += ['',
        f'Observed mean difference between 50 and 5 demonstrations: {(means[-1]-means[0])*100:.1f} percentage points.',
        'This is a descriptive comparison at a fixed training budget. It cannot establish where the curve plateaus',
        'or how many real-robot demonstrations would be needed.',
        'Figure: left, mean with two-way bootstrap interval; right, each model with its Wilson 95% interval.',
        'Models share test cases, so the 150 outcomes per demonstration count are not pooled into one Wilson interval.', '',
        'Checks: all 15 final checkpoint hashes differ; demonstration subsets, frame counts and normalization sample counts',
        'are correct; every run reached 6,000 updates with finite logged loss; all 750 test-case IDs are present;',
        'per-episode outcomes match the official aggregates.',
        f'An audit replay of the same checkpoint and seed differed from the first evaluation in {difference_count}/750 success outcomes',
        f'and {changed_steps}/750 step counts. All success rates here use the first formal evaluation; replays are stored',
        'separately and were never used to select results or counted as extra samples.',
        'Initial qpos of the 50 test cases match within 1e-8 across all 15 replays. These poses are reconstructed from',
        'same-seed replays, not logged by the first run; initial_qpos_source in the merged records says so.',
        'The formal evaluation saved videos of the first two test cases per model (30 in total); replays have their own.', '',
        'Pilot: separate training seed 777 and test seeds 20000-20019; 2k/4k/6k updates gave 10/20, 16/20, 19/20.',
        'Formal training seeds 0/1/2, test cases 10000-10049. Neither the pilot nor the replays add formal samples.',
        'The shared update count was frozen before formal training; only final checkpoints were evaluated.', '',
        'Logging fix: the original records overwrote the initial pose at SAME_STEP autoresets. The original data is kept;',
        'only the pose-logging point changed, replaying the same checkpoint on the same test cases. Training and the',
        'official rollout are unchanged.', '',
        'Repeatability limit: the official evaluator enables cuDNN benchmark and strict determinism was not enforced.',
        'Replays of the same weights and seed differ slightly; the numerical source was not fully isolated. The',
        'difference is reported separately and is not modelled by the bootstrap interval.', '',
        'Infrastructure rerun: the original N=20, seed=2 instance ran about twice as slow and would have exceeded its',
        'time limit. It was stopped before any evaluation, its logs kept, and the same configuration and seed rerun on',
        'a fresh A10. 15 complete models plus 1 incomplete attempt; the incomplete attempt is not scored as 0 and no',
        'rerun was chosen by evaluation outcome.', '',
        f'Conservative resource estimate for this experiment: ${estimated:.2f} in total; formal training and first evaluation',
        f'${formal_cost:.2f}, audit replay ${audit_cost:.2f}. Includes speed tests, debugging, the pilot and retained cancelled',
        'reservations. Not a Modal invoice; CPU builds, storage and transfer are extra. Budget ceiling $15.', '',
        'Limitations: 3 training seeds; one nested subset chain; the bootstrap can be too narrow near 0% or 100%;',
        '6,000 fixed updates do not guarantee every group has converged; scripted demonstrations in simulation do not',
        'transfer directly to human teleoperation or real hardware.', '',
        'Files: summary.csv, per_model.csv, episodes.jsonl, audit.json, statistics.json, data_scaling.png/svg/pdf.',
        'Code, exact configuration and reproduction notes: ../README.md.',
    ]
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
