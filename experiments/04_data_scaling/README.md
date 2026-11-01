# Experiment 4: how many demonstrations does ACT need?

**Question.** As the number of training demonstrations grows from 5 to 50, how does task success change? This is a small simulated version of a demonstration-budget study, using ACT because it trains quickly.

**Result.** 15 models (5 data sizes × 3 training seeds), each evaluated on the same 50 test cases: 750 formal outcomes.

| Demonstrations | Seed 0 | Seed 1 | Seed 2 | Mean | Bootstrap 95% CI |
|---|---|---|---|---|---|
| 5 | 12% | 34% | 22% | 22.7% | 9.3–38.0% |
| 10 | 56% | 36% | 48% | 46.7% | 31.3–62.0% |
| 20 | 78% | 80% | 48% | 68.7% | 48.0–86.0% |
| 35 | 94% | 86% | 88% | 89.3% | 80.0–97.3% |
| 50 | 84% | 86% | 90% | 86.7% | 77.3–94.7% |

Success climbs from 23% to 89% between 5 and 35 demonstrations, with no further gain at 50 for this training budget. With three seeds this describes the curve; it does not pin down where it plateaus.

## Design

- Task: `AlohaTransferCube-v0` from gym-aloha 0.1.5. Success means reward = 4.
- Data: `lerobot/aloha_sim_transfer_cube_scripted` at revision `c5467ba9059e4c65f27775ced212076980a173fb`: 50 scripted demonstrations of 400 frames each. Scripted data was chosen because a public community run on the human demonstrations reached only about 42% with all 50 episodes.
- Model: ACT with a generic ImageNet ResNet18 backbone; no task-specific pretrained policy. Defaults otherwise: fixed learning rate 1e-5 (this ACT version has no scheduler option), `chunk_size = n_action_steps = 100`, `latent_dim = 32`, KL weight 10.
- Subsets: one seed-0 permutation of the 50 demonstrations; the first N form each subset, so smaller subsets are nested in larger ones (`configs/rq1_subsets.yaml`).
- Training: each model starts from a fresh initialization, trains for exactly 6,000 updates at batch size 8 on one A10, and only the final checkpoint is evaluated. State/action normalization statistics come from that model's subset only; image normalization uses fixed ImageNet constants.
- Evaluation: environment seeds 10000–10049 for every model; videos of the first two test cases per model, chosen in advance.
- Update count: frozen before formal training from a separate 50-demonstration pilot (training seed 777, test seeds 20000–20019): 10/20 after 2,000 updates, 16/20 after 4,000, 19/20 after 6,000. The pilot is not part of the results.

Exact settings: `configs/rq1_plan.json`, `configs/final_design.json`, `analysis_plan.json`.

### Data loading

The dataset's AV1 videos were decoded once on CPU into lossless RGB PNG frames (`build_data.py`). All 20,000 frame timestamps were checked, and seven sampled frames matched the official PyAV decode pixel for pixel. Only video reading was replaced; LeRobot's sampling, action chunking and padding are unchanged. Every training run checks its episode IDs and frame count (N × 400) in `subset_audit.json`. Two compatibility patches register `gym_aloha` for evaluation subprocesses and add a `task_description` attribute; ACT does not read language, and neither patch touches physics, actions or rewards.

### Deviations

Recorded in `execution_deviations.json`; the frozen design file was not edited.

- **Initial-pose logging.** `evaluate_act.py` recorded the initial joint positions after every reset, and LeRobot's same-step autoreset overwrote them at the end of each episode. `audit_evaluate.py` records them only at explicitly seeded resets. `modal_audit.py` replayed every final checkpoint on the same 50 test cases to recover the poses. The first formal evaluation stays the primary result. The replays changed 3 of 750 outcomes and 51 of 750 step counts (`results/replay_comparison.csv`); the official evaluator enables cuDNN benchmark and strict determinism was not enforced, and the exact numerical source was not isolated. Replays are not counted as extra samples.
- **Slow instance.** The original N=20, seed 2 instance trained at about 0.20 s per update instead of 0.09 s and would have exceeded its 1,200 s limit. It was stopped at update 2,400, before any evaluation, and its logs are in `attempts/`. The same configuration and seed were rerun from a fresh initialization on another A10 with a 1,800 s limit (`modal_retry.py`). It is not scored as a failure, and no rerun was chosen by outcome.

## Statistics

- Main curve: arithmetic mean over the three training seeds.
- Band: two-way bootstrap that resamples training seeds and the shared test cases together, 10,000 replicates, seed 20261030.
- Each model also gets a Wilson 95% interval over its 50 outcomes.
- The 150 outcomes per data size share test cases, so they are not pooled into one Wilson interval.

## Files

| File | Content |
|---|---|
| `modal_act.py` | Modal images, budget ledger and entry point; modes `cpu-check`, `eval-check`, `prepare`, `smoke`, `pilot`, `formal` |
| `train_act.py`, `evaluate_act.py` | Training and evaluation wrappers around LeRobot (hashes recorded in every run manifest) |
| `build_data.py`, `discover_dataset.py` | Lossless frame cache and dataset discovery |
| `modal_audit.py`, `audit_evaluate.py` | Pose-recovery replay of the final checkpoints |
| `modal_retry.py` | The one infrastructure replacement run |
| `fetch_results.py`, `verify_videos.py` | Download and video audit |
| `summarize.py` | Validator: checks configs, subsets, losses, checkpoints, seeds and replays against the raw run folders, then writes `results/` |
| `configs/` | Frozen plan, subsets and final design |
| `analysis_plan.json`, `execution_deviations.json`, `data_audit.json`, `cpu_check.json`, `audit_recovery.json` | Plan, deviations and audits |
| `budget_ledger.json`, `audit_budget_ledger.json`, `retry_budget_ledger.json`, `audit_overhead_ledger.json` | GPU reservations and settled estimates (the main ledger includes the others) |
| `attempts/` | Logs of the stopped slow instance |
| `results/episodes.jsonl` | 750 records: model, seed, test case, success, steps, replay outcome, initial pose and its source |
| `results/summary.csv`, `per_model.csv`, `statistics.json`, `audit.json`, `replay_comparison.csv`, `report.txt` | Results, audits and the full report |
| `results/runs/` | The 15 formal run manifests (commands, hashes, timings, official metrics) |
| `results/data_scaling.*`, `example_rollouts.jpg` | The figure as first produced; example frames |

## Rerunning

The commands used, from this folder:

```bash
modal run modal_act.py --mode prepare
modal run modal_act.py --mode formal
modal run modal_audit.py
python fetch_results.py --all-completed
python fetch_results.py --all-completed --audit
```

Before a rerun outside our workspace:

- `modal_act.py` loads the CPU-built frame cache from our Modal image ID. Replace the line `data_image = modal.Image.from_id(...)` with `data_image = data_image_recipe` to rebuild it from the public dataset.
- `modal_act.py` reserves every GPU job against `budget_ledger.json` with a $13.50 ceiling. Start from an empty ledger (move the four ledger files away) or the reservation check will refuse new jobs.
- `summarize.py` expects the raw run folders under `modal_results/` and `audit_results/`.

Final checkpoints are kept in our private Modal volume `knightec-vla-stage3b` and are not published. Each run manifest gives the checkpoint path and its `model.safetensors` SHA-256. Total resource estimate for this experiment, including speed tests, debugging, the pilot and the audit replays: $9.40 (run time × list price, not an invoice).

## Limitations

- Three training seeds and one nested subset chain: the effect of drawing a different set of demonstrations is not measured.
- The bootstrap is descriptive with so few seeds and can be too narrow near 0% or 100%.
- A fixed 6,000 updates compares data sizes at equal compute; it does not show that every model has converged or find the minimum demonstrations needed.
- Scripted demonstrations in simulation do not transfer directly to human teleoperation or real hardware.
