# Experiment 1: π0.5 baseline on LIBERO-Goal

**Question.** Does our evaluation pipeline reproduce LeRobot's published π0.5 result? Every later π0.5 number is compared against this run.

**Result.** 98/100 episodes succeeded (98.0%, Wilson 95% CI 93.0–99.4%). LeRobot reports 98.0% for LIBERO-Goal ([LeRobot LIBERO docs](https://huggingface.co/docs/lerobot/en/libero)), so the difference is 0.0 points.

| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Successes / 10 | 10 | 10 | 9 | 9 | 10 | 10 | 10 | 10 | 10 | 10 |

Evaluation took 1,039 s on one NVIDIA A100-SXM4-40GB (10.4 s per episode), or 1,221 s including model loading.

## Setup

- Model: `lerobot/pi05_libero_finetuned_v044` at revision `7152f13e34c6c115d808aea0e5d08aca74eac14b`
- LeRobot commit `72e1d9e5e76ee719bacfc78bc34d817bee5fb521`, torch 2.11.0, built into a Modal image (`modal_pi05.py`)
- `lerobot-eval` unchanged: batch size 1, hard reset, the suite's fixed initial states (`--env.init_states=true`), relative control, `n_action_steps=10`, BF16, model compilation off, seed 0
- 10 Goal tasks × 10 episodes

Model weights and LIBERO assets are downloaded once on a CPU container (`prepare`) into the Modal volume `knightec-vla-stage1`. Experiments 2 and 3 reuse that cache.

## Files

| File | Content |
|---|---|
| `modal_pi05.py` | Modal image and entry point; modes `smoke`, `goal`, `baseline` (all four suites) |
| `fetch_results.py` | Download one run folder from the Modal volume |
| `summarize_result.py` | Turn `eval_info.json` into per-episode records, per-task table and report |
| `results/run.json` | Run manifest: model, commits, GPU, exact command, official metrics, log tail |
| `results/eval_info.json` | LeRobot's own evaluation output |
| `results/episodes.jsonl` | One record per episode (task, episode, success) |
| `results/task_summary.csv`, `summary.json`, `run-report.txt` | Per-task and overall results |
| `results/packages.txt` | `pip freeze` of the evaluation container |

## Rerunning

```bash
modal run modal_pi05.py --mode smoke --max-seconds 1200          # 2 LIBERO-Object episodes
modal run modal_pi05.py --mode goal --seed 0 --max-seconds 3600  # this experiment
python fetch_results.py goal_seed0_<timestamp>
python summarize_result.py goal_seed0_<timestamp>
```

Requires a Modal secret `knightec-huggingface` with `HF_TOKEN`. `--mode baseline` runs all four LIBERO suites (400 episodes); it was not used.

## Limitations

- Goal suite only, one evaluation seed. The other three LIBERO suites and the planned three-seed average were not run.
- Wilson intervals treat each episode as an independent Bernoulli trial.
