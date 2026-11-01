# Experiment 3: does π0.5 read the instruction?

**Question.** If we change only the instruction, does π0.5 still complete the *original* task? If success barely moved, the policy would be acting from the scene and initial state alone. This is the "swap the instruction and measure what happens" test.

**Result.**

| Condition | Instruction sent to the policy | Original-task success | Wilson 95% CI | Lost / gained vs. C0 | Exact McNemar p | Holm-adjusted |
|---|---|---|---|---|---|---|
| C0 original | the task's own instruction | 97/100 | 91.5–99.0% | — | — | — |
| C1 paraphrase | first published LIBERO-PRO paraphrase | 97/100 | 91.5–99.0% | 1 / 1 | 1.0 | 1.0 |
| C2 swap | the next task's instruction | **0/100** | 0–3.7% | 97 / 0 | 1.3 × 10⁻²⁹ | 5.0 × 10⁻²⁹ |
| C3 empty | empty task string | 11/100 | 6.3–18.6% | 86 / 0 | 2.6 × 10⁻²⁶ | 5.2 × 10⁻²⁶ |
| C4 unrelated | "The weather forecast predicts rain tomorrow." | 8/100 | 4.1–15.0% | 89 / 0 | 3.2 × 10⁻²⁷ | 9.7 × 10⁻²⁷ |

π0.5 follows the instruction it is given. Rewording does not matter, but naming a different task, or giving no usable task, stops it from completing the original one. `inspection/paired_instruction_example.png` shows one pair: with the original instruction the robot opens the drawer; with "put the bowl on the stove" it puts the bowl on the stove, from an identical start.

## Design

- Same checkpoint, LeRobot commit, image and rollout settings as [experiment 1](../01_baseline/): BF16, compilation off, `n_action_steps=10`, batch size 1, relative actions, hard reset, the suite's initial states, run seed 0. No training.
- LIBERO-Goal: 10 tasks share one scene and differ only in their goal, so the instruction is the only cue to which task is meant.
- 10 tasks × 10 initial states = 100 pairs per condition, 500 formal episodes.
- **C1** comes from LIBERO-PRO's `libero_ood/ood_language.yaml` at commit `eafdb809…` (the source of its language-perturbed BDDL files); `prepare_prompts.py` takes the first paraphrase per task and stores all candidates in `configs/paraphrases.json`.
- **C2** maps task *i* to task (*i*+1) mod 10 in the benchmark's task order.
- **C3** is an empty task string, not removal of all language tokens: π0.5 still receives its fixed `Task: …, State: …; Action:` template and its proprioceptive state tokens.
- Success is always judged by the original task's goal predicate. Whether the robot completed the swapped-in task was not measured.

### Custom rollout loop and pairing

`run_swap.py` is an explicit batch-one rollout loop. It reuses the pinned LeRobot environment and model setup, processors, metrics and video writing, and follows LeRobot's Apache-2.0 rollout (relative-action anchors, resets, action queue, success extraction). The only change is that `observation['task']` is replaced before the policy preprocessor. The environment's task metadata and success predicates are unchanged.

The C0 run saves, for every episode, the supplied and settled simulator-state hashes, hashes of the visual model parameters (cameras, lights, colors), the first camera images and the CPU/CUDA torch RNG state. Every other condition then:

1. asserts identical physical state and visual-model parameters,
2. checks the first images within a fixed tolerance,
3. restores that pair's RNG state before the first action, since π0.5's sampling noise comes from `torch.normal`,
4. records the formatted prompt and first-token hash, and asserts that the task tokens changed while the state-token text did not.

All 400 intervention pairs passed these checks (`results/summary.json` → `pairing_audit`).

### Gates, deviations and how they were handled

These steps were all fixed or disclosed before any formal intervention outcome was seen:

- **Image tolerance.** Two independent reset probes matched physics and visual parameters exactly, but a few color channels differed by one intensity level. The tolerance was set to at most 1/255 difference in at most 0.1% of channels (`diagnostic_result.json`). Later full reset-only preflights across cloud platforms found rare 2/255 differences and post-reset state differences at machine precision (5.2 × 10⁻¹⁸). The final validation overlay allows 2/255 in at most 0.1% of channels and aligns physical states to the C0 snapshots only when the largest difference is ≤ 10⁻¹² (`validation_overrides.json`). No images or actions were modified; raw images and per-pair differences are saved.
- **Control gate.** C0 was first required to match experiment 1 episode for episode. It differed in one episode (task 2, initial state 5). An equivalence check (`loop_equivalence_result.json`) then replayed real observations through the official loop and the custom loop: actions, rewards and done flags were exactly equal across 195 steps. The gate was amended to: that equivalence check, an absolute success-rate difference ≤ 5 points, and exact McNemar p ≥ 0.05 against experiment 1. C0 passed (97 vs. 98, p = 1.0). The original C0 run was kept, not rerun for a better outcome. This is an engineering compatibility check, not proof of equivalence (`control_validation.json`).
- **Interrupted C3.** The first C3 attempt completed 39 pairs before the image guard stopped it. Only the missing 61 pairs were resumed, with matching seed and state offsets; `merge_resume.py` joins the two non-overlapping parts and maps each video to its true episode. No completed episode was replaced.
- C1–C4 then ran in parallel, one A100 each, with a one-hour limit and no automatic retries: 172 GPU-minutes in total for the runs that contributed formal records.

The primary comparisons (C2 and C3 against C0, exact two-sided McNemar, Holm adjustment over all four) were fixed in `analysis_plan.json` before formal outcomes.

## Files

| File | Content |
|---|---|
| `run_swap.py` | Paired rollout loop with instruction replacement and pairing checks |
| `audited_runner.py` | Validation wrapper applying the tolerance overlay; `source_history/` keeps its earlier versions |
| `modal_rq2.py` | Modal image and entry point; modes `prepare`, `smoke`, `control`, `variants`, `diagnose`, `equivalence`, `validate-control`, `recover-states`, `all` |
| `prepare_prompts.py` | Extract paraphrases from a LIBERO-PRO clone |
| `probe_reset.py`, `preflight_states.py`, `snapshot_control.py`, `check_loop.py` | Reset probes, state preflight, C0 snapshots, loop-equivalence check |
| `merge_resume.py`, `fetch_results.py`, `verify_videos.py`, `inspect_pair.py` | Merge the resumed C3, download, video/step audit, paired frame figure |
| `make_plots.py` | Validator: re-checks every pairing hash, image tolerance, prompt, video count and official metric against the raw run folders, then writes `results/` |
| `configs/rq2_swap.json` (= `.yaml`), `configs/paraphrases.json` | Frozen protocol: all 50 instructions, pairing, tolerances |
| `analysis_plan.json`, `control_validation.json`, `validation_overrides.json`, `diagnostic_result.json`, `loop_equivalence_result.json`, `control_reference.json`, `control_states_result.json` | Pre-registered analysis, gate decisions and their evidence |
| `results/episodes.jsonl` | 500 records with instruction, formatted prompt, state/image/RNG hashes, steps, success |
| `results/summary.csv`, `paired_comparisons.csv`, `per_task.csv`, `summary.json`, `run-report.txt` | Results, paired tests, per-task counts |
| `results/C*.json`, `artifact_audit.json` | Run manifests and video audit |
| `inspection/` | Paired example frames |

## Rerunning

The commands used, from this folder:

```bash
python prepare_prompts.py                                   # needs a LIBERO-PRO clone at the pinned commit
modal run modal_rq2.py --mode all --max-seconds 3600         # prepare → smoke → control → variants
python fetch_results.py <run folder name>
python merge_resume.py <resumed C3 run folder name>          # only if a condition was resumed
```

Before a rerun outside our workspace, note that:

- the control gate compares against experiment 1's run by name (`/data/results/goal_seed0_20261028T233602Z` in `modal_rq2.py`); point it at your own baseline run;
- `--mode all` rewrites `configs/rq2_swap.json`, `control_reference.json` and, through `validate-control`, `control_validation.json` in place, so work on a copy or a branch;
- `make_plots.py` expects the raw run folders (videos, initial images, RNG snapshots) under `modal_results/` and `control_states/`, and should run on Linux or WSL (it reads JSON with the platform default encoding).

## Limitations

- One run seed, ten tasks, ten initial states per task. Episodes of the same task may be correlated; Wilson and McNemar do not model this clustering.
- Only the original task is scored. A failure after a swap does not show that the robot completed the swapped-in task, although the inspected example did.
- One fixed cyclic swap, one paraphrase per task, one unrelated sentence.
- Changing the task text also changes the token count and the positions of the state tokens. The experiment measures sensitivity to the task text; it does not separate semantic understanding from formatting effects.
