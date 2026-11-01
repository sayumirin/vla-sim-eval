# Experiment 2: visual perturbations with LIBERO-plus

**Question.** How much does π0.5's LIBERO-Goal success drop under changes in lighting, camera viewpoint and object layout?

**Result.**

| Perturbation | Success | Wilson 95% CI | Difference from 98/100 baseline | Mean over 10 base tasks |
|---|---|---|---|---|
| Lighting | 100/100 | 96.3–100% | +2 pp | 100% |
| Camera viewpoint | 74/100 | 64.6–81.6% | −24 pp | 75.3% |
| Object layout | 71/100 | 61.5–79.0% | −27 pp | 70.3% |

Lighting changes did not hurt. Camera and layout changes each cost about a quarter of the successes.

## Setup

- Same checkpoint, LeRobot commit and rollout settings as [experiment 1](../01_baseline/).
- LIBERO-plus `sylvestf/LIBERO-plus` @ `4976dc30028e805ff8094b55501d532c48fec182`, installed in a separate image that replaces `hf-libero` and uses robosuite 1.4.1 (the baseline used 1.4.0). Asset archive revision and SHA-256: `audit/assets_manifest.json`.
- LIBERO-Goal variants only. For each perturbation category (`Light Conditions`, `Camera Viewpoints`, `Objects Layout`), 100 distinct variants were sampled uniformly without replacement (sampling seed 0), one episode each (evaluation seed 0), hard reset, supplied initial states.
- The three categories ran in parallel, one A100 each: 76 GPU-minutes in total including model loading. Three smoke episodes ran before the formal runs and are excluded from the results.

### Sampling

`select_plus_tasks.py` reads LIBERO-plus's `task_classification.json`, maps its one-based IDs to the zero-based indices the LeRobot CLI expects by exact task name, and writes all 300 selections to `configs/plus_tasks.json`. `configs/plus_tasks.yaml` has the same content (JSON is valid YAML 1.2). `audit/selection_audit.json` records every mapping.

### Protocol adjustment: prompts

The pinned LeRobot adapter builds the language prompt from the variant's file name, which would add perturbation suffixes such as `_light_12` to the instruction. `run_plus_eval.py` replaces each prompt with the original base-task instruction, matching experiment 1. It does not change actions, weights, resets, simulation or success checks. Raw and normalized prompts are in `audit/selection_audit.json`. This is a documented custom protocol, not the unmodified LIBERO-plus CLI.

### Failure inspection

Two failures were inspected frame by frame (`inspection/`):

- Layout variant 1928 (`open_the_middle_drawer_of_the_cabinet_moved_level5_sample2`): the scene renders normally and the arm moves, but the drawer is still closed at the end.
- Camera variant 762 (`put_the_wine_bottle_on_top_of_the_cabinet_view_35_15_100_0_0_initstate_0`): the view is visibly tilted, the arm works near the bottle, but the bottle is still on the table at the end.

Four frames confirm the scenes rendered correctly and the goal was not reached; they do not explain why the policy failed.

## Files

| File | Content |
|---|---|
| `modal_plus.py` | Modal image and entry point; modes `prepare`, `smoke`, `full`, option `--category` |
| `run_plus_eval.py` | Prompt normalization and step auditing around `lerobot-eval` |
| `select_plus_tasks.py` | Seeded variant sampling (needs a LIBERO-plus clone at the pinned commit) |
| `check_smoke.py`, `fetch_results.py`, `fetch_audit.py`, `audit_parallel.py`, `inspect_video.py` | Smoke-run checks, download helpers, parallel-run audit, frame extraction |
| `make_plots.py` | Validator: checks hashes, task IDs, videos and per-episode outcomes against the raw run folders, then writes `results/` |
| `configs/` | The 300 sampled variants |
| `audit/` | Asset manifest, selection audit, smoke checks, parallel-run record |
| `results/episodes.jsonl` | 300 records: category, variant, base task, difficulty, prompt, steps, success |
| `results/summary.csv`, `per_base_task.csv`, `summary.json`, `run-report.txt` | Summaries, runtime and resource estimate |
| `results/{light,camera,layout}.json` | Full run manifests including official metrics |
| `results/robustness.png/.pdf` | The figure as first produced |

## Rerunning

From this folder, after experiment 1 has created the model cache:

```bash
modal run modal_plus.py --mode prepare
modal run modal_plus.py --mode smoke --max-seconds 600
modal run modal_plus.py --mode full --max-seconds 3600              # or --category light|camera|layout per GPU
python fetch_results.py <run folder name>
```

To run the three categories in parallel, start three `--category` commands instead of one `--category all`; otherwise the same samples run twice. Each run has no automatic retries and a one-hour limit including model loading. Outcomes are appended to `progress.jsonl` as each variant finishes. `make_plots.py` expects the raw run folders under `modal_results/` and the baseline under `../stage1/` (its original location), so it does not run from a fresh clone.

The formal runs plus smoke tests cost an estimated $3.62 (A100 run time × Modal list price). This is not an invoice.

## Limitations

- A sample of the LIBERO-plus Goal variants, one episode per variant, one seed. Not the full published benchmark.
- Base-task and difficulty mixtures differ between categories and from the balanced baseline (see `per_base_task.csv`), so differences from the baseline are not paired comparisons.
- Wilson intervals do not model correlation between related variants or seed and sampling uncertainty.
- robosuite 1.4.1 here vs. 1.4.0 in the baseline.
- "Object layout" combines moved objects and added distractors; it does not measure distractors alone.
- Some upstream lighting variants have no difficulty label; they are kept as unknown.
