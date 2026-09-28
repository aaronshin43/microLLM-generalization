# Detection Baselines: Provenance and Reproduction

This record connects the [final report](../FINAL_REPORT.md) to existing local artifacts. It separates **saved-result inspection**, **short pipeline validation**, and **full experiment reproduction**. Only the first two were performed during this cleanup; no training-length sweep or 10M evaluation was run. Read [DETECTION_GUIDE.md](DETECTION_GUIDE.md) for the model and [PLAN.md](PLAN.md) for the follow-up scope.

Paths below are relative to `infinite_generalization/`. Run artifacts are ignored by the existing Git policy; they have not been forced into tracking. These links work in this local checkout but do not establish availability in a fresh clone.

## Report baselines and saved results

The report specifies five seeds (0–4). `src/make_report_figures.py` explicitly reads `runs/stage3_seeds/<condition>_s<seed>/metrics_by_length.csv`, linking the six Figure 1 conditions to that family. All eight Table 1 conditions have five existing run directories containing `config.json`, `train_history.csv`, `metrics_by_length.csv`, and `model.pt`.

The table below was recalculated from those saved CSVs at 10M, **without reevaluating 10M sequences**. Continuous columns show mean ± sample standard deviation (`statistics.stdev`, five seeds); accuracy is positive accuracy and identical across seeds. The values round to the report's Table 1. Products are computed per seed before aggregation, not by multiplying aggregate means.

| Condition / seed 0 artifact directory | Epochs | Updates | $\Delta$ | $c$ | $c\Delta$ | Target mass | Positive logit | Positive accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [constant_e50](../../runs/stage3_seeds/constant_e50_s0/) | 50 | 1600 | 8.977 ± 0.217 | n/a | n/a | 0.000805 ± 0.000162 | -3.258 ± 0.074 | 0% |
| [constant_e100](../../runs/stage3_seeds/constant_e100_s0/) | 100 | 3200 | 9.882 ± 0.233 | n/a | n/a | 0.001995 ± 0.000427 | -4.544 ± 0.063 | 0% |
| [constant_e1000](../../runs/stage3_seeds/constant_e1000_s0/) | 1000 | 32000 | 13.161 ± 0.237 | n/a | n/a | 0.050310 ± 0.010417 | -17.303 ± 0.413 | 0% |
| [log_e50](../../runs/stage3_seeds/log_e50_s0/) | 50 | 1600 | 4.391 ± 0.145 | n/a | n/a | 1.000000 ± 0.000000 | 3.244 ± 0.073 | 100% |
| [learned_log_e50](../../runs/stage3_seeds/learned_log_e50_s0/) | 50 | 1600 | 8.097 ± 0.236 | 0.0720 ± 0.0057 | 0.5819 ± 0.0336 | 0.788331 ± 0.074664 | 1.923 ± 0.486 | 100% |
| [learned_log_e100](../../runs/stage3_seeds/learned_log_e100_s0/) | 100 | 3200 | 8.631 ± 0.258 | 0.0963 ± 0.0075 | 0.8302 ± 0.0466 | 0.996855 ± 0.001572 | 4.570 ± 0.063 | 100% |
| [learned_log_e200](../../runs/stage3_seeds/learned_log_e200_s0/) | 200 | 6400 | 9.034 ± 0.275 | 0.1259 ± 0.0099 | 1.1354 ± 0.0635 | 0.999983 ± 0.000012 | 6.415 ± 0.056 | 100% |
| [learned_log_e400](../../runs/stage3_seeds/learned_log_e400_s0/) | 400 | 12800 | 9.386 ± 0.293 | 0.1658 ± 0.0129 | 1.5536 ± 0.0841 | 1.000000 ± 0.000000 | 9.586 ± 0.054 | 100% |

For seeds 1–4, replace `_s0` with the seed suffix. All 40 recorded negative accuracies at 10M are 100%. Learned-log e50/e100 are below the growth threshold; e200/e400 are above it in each stored seed. This demonstrates why passing the finite benchmark is different from having target mass converge to 1.

### Verified configuration

The original machine-readable run configs exist. The commands later in this document are **reconstructed from those configs**, not original shell transcripts. Common settings are:

```text
train_lengths: [10]
target_position_mode: fixed_start
target_token_count: 1
non_target_token_count: 1
non_target_sampling: uniform
train_examples: 2000 (per length)
val_examples: 500 (per length)
test_examples: 50 (per evaluation length, 25 positive + 25 negative)
eval_lengths: 10 100 1000 10000 100000 1000000 10000000
eval_chunk_examples: 50
eval_sampling_mode: random
batch_size: 64
eval_batch_size: 16
learning_rate: 0.003
weight_decay: 0
d_head: 2
alpha_log_scale_init: -5
max_train_steps: null
device: auto; resolved_device: cuda
seed: 0, 1, 2, 3, 4
```

For every run, history's last update count agrees with config and checkpoint metadata, and equals epochs × 32. On CPU at length 10, checkpoint-derived margin and positive forward logit agree with saved CSV values within `rtol=3e-6`, `atol=3e-5`. This allows small float32 reduction differences across the historical CUDA execution and current CPU inspection; it is not a tolerance claim for explicit 10M forward evaluation.

The historical code revision and exact Python/PyTorch/GPU versions are absent from the run metadata. Current revision and environment below describe this cleanup only. Existing artifacts and matching configs establish a reproducible specification, but do not establish that current code and environment will regenerate identical trained weights.

### Figure provenance

- Figure 1: [make_report_figures.py](../../src/make_report_figures.py) reads the six `RUN_SPECS` conditions: constant e50/e100/e1000, log e50, learned-log e50/e200, over seeds 0–4. It uses `mean_empirical_target_attention` and `mean_logit_positive`, and sample standard deviation. Its output is `documents/latex/final_report_attention_and_logit_by_length.pdf` and `.png`.
- Figure 2: [make_mechanism_figure.py](../../src/make_mechanism_figure.py) draws **hardcoded rounded vectors**, rather than reading a checkpoint. Its specified learned-log e200 seed 1 vectors match the existing checkpoint rounded to three decimals: $q_u=(-1.830,1.646)$, $k_t=(-2.232,1.642)$, $k_u=(1.990,-1.428)$, with $\Delta\approx9.0358$. Outputs are in `documents/figures/` and `documents/latex/`.
- Both scripts contain absolute paths for this Windows checkout. Both accept `--preview` for separate preview filenames. They were inspected but not executed; existing report figures were preserved. `make_report_figures.py` loads data at module import, so even importing it reads the baseline CSVs.

### Historical families kept separate

`runs/stage3base/learned_log_e200/` exists with config, metrics, history and figures, but **no `model.pt`**. Its config still records the former output directory `runs/stage3_simplified_attention_learned_log_e200`; use the actual artifact location, not that stale metadata path. No original code revision is recorded. Do not attribute the report's five-seed results to this seed 42 family.

[STAGE3_WEIGHT_LEVEL_MECHANISM.md](../STAGE3_WEIGHT_LEVEL_MECHANISM.md) explicitly names `runs/stage3_mechanistic_interpretation/learned_log_e200/`. Its stored config describes a separate seed 42 run, 6400 updates, eval batch size 8; it is not report seed 1. [Stage 3 / 3B](../STAGE3_SIMPLIFIED_LENGTH_AWARE_ATTENTION.md) retains historical single-seed results and existing multi-length exploration. Different numerical margins there are not automatically documentation errors.

The existing Stage 3B directory includes `stage3b_single_length_learned_log_steps6400`, `stage3b_multilength_learned_log_10_20_50_steps6400`, `stage3b_multilength_learned_log_10_20_50_100_steps6400`, `stage3b_multilength_learned_log_10_100_1000_10000_100000_steps6400`, and `seed123_logspaced_steps6400`. These are historical comparison candidates; their full numerical reproduction was not performed here. The Stage 3B findings remain evidence from those tested conditions, not a universal law about mixtures.

## Short validation: actually executed

Validation output was placed in the new, initially nonexistent directory:

```text
runs/training_length/cleanup_checks/20260928_detection_cleanup_01/
```

Run commands below from the repository root in PowerShell, using the existing `.venv`. The actual smoke loop also captured console output to `smoke_<mode>.txt` and stopped on nonzero exit. Repeating this exact output path would overwrite the cleanup smoke artifacts; choose a new unique directory for a later check.

```powershell
Set-Location D:\03_Coding\microLLM-generalization
& ./.venv/Scripts/python.exe infinite_generalization/src/stage3_simplified_attention.py --help
& ./.venv/Scripts/python.exe infinite_generalization/runs/training_length/cleanup_checks/20260928_detection_cleanup_01/audit_baselines.py
& ./.venv/Scripts/python.exe -m unittest discover -s infinite_generalization/tests -p 'test_stage3*.py' -v
foreach ($checkMode in @('constant', 'log', 'learned_log')) {
    & ./.venv/Scripts/python.exe infinite_generalization/src/stage3_simplified_attention.py --smoke-test --device cpu --seed 42 --alpha-mode $checkMode --test-examples 12 --eval-chunk-examples 5 --output-dir "runs/training_length/cleanup_checks/20260928_detection_cleanup_01/smoke_$checkMode"
    if ($LASTEXITCODE -ne 0) { throw "Smoke check failed: $checkMode" }
}
```

`--smoke-test` uses 64 train / 32 val examples per active length, batch size 16, 2 epochs, eval lengths 10 and 20, and eval batch size 20. Here it made **8 updates per mode**, evaluating 12 examples per length in chunks of at most 5. CLI `train_lengths`, `test_examples`, chunk size and `max_train_steps` survive smoke overrides; do not assume it caps every user-supplied setting. These tiny runs check training, saving and evaluation, not learned length generalization or reproduction of the original result.

| Validation | Result | Evidence in cleanup directory |
|---|---|---|
| Saved baseline inspection | 40 configs, histories and checkpoints checked | `baseline_inventory.json`, `baseline_summary.json` |
| Historical geometry | Report seed 1 vectors match checkpoint rounding | See figure provenance above |
| Stage 3 evaluation tests | 11 passed | `stage3_tests.txt` |
| Independent two-score calculation | All 3 modes passed, positive readout/logit and negative output/logit | `closed_form_checks.json`, `audit_baselines.py` |
| Short pipeline | All 3 modes completed; 8 updates, metrics at 10/20 | `smoke_<mode>/config.json`, history, metrics, checkpoint, figures |

Final checks also verified all common settings above across the 40 configs,
the learned-log threshold side for each seed, 37 local Markdown link targets,
the three saved smoke budgets, and Git whitespace checks. No missing local link
target or whitespace error was found.

The independent calculation uses scalar Python math rather than the model's theory helper, on the controlled length-10 example in the guide. `rtol=2e-6`, `atol=2e-7` allow float32 rounding for this small reduction. No forward/scoring/gradient implementation was changed, so no new gradient comparison or Stage 4 retraining was needed. Existing evaluation tests do not by themselves cover the full training procedure; smoke runs provide the limited pipeline check.

Environment recorded in `environment.json`:

```text
Cleanup date: 2026-09-28
Git revision: e816c36b998bbf808f439ba365b148b5ea098e32
Initial tracked dirty state: M .gitignore (user change, preserved)
Python: 3.12.10, Windows AMD64
PyTorch: 2.12.0+cu126
CUDA runtime reported by PyTorch: 12.6
Available GPU: NVIDIA GeForce RTX 4060 Laptop GPU
Audit and smoke device: CPU
Virtual environment: repository-root .venv
```

No dependencies were upgraded, no virtual environment created, no commit/push/activity log written. Existing runs, checkpoints, configs, report figures and research claims were not modified.

## Full training reproduction: not executed

Stage 3 has no YAML config loader. `config.json` is saved metadata, not an accepted `--config` input. No new config system or empty shared config directory was created. The following CLI command restores report learned-log e200 seed 0 settings under the **current** implementation. It includes expensive 10M evaluation and is not a smoke command.

First choose an output directory that does not exist. In this example, run from `infinite_generalization/` after the README environment setup:

```powershell
Set-Location D:\03_Coding\microLLM-generalization\infinite_generalization
$env:PYTHONPATH = "src"
$baselineOutput = "runs/training_length/report_reproduction/learned_log_e200_s0_new"
if (Test-Path $baselineOutput) { throw "Choose a new output directory." }
& ../.venv/Scripts/python.exe -m stage3_simplified_attention --seed 0 --device auto --output-dir $baselineOutput --alpha-mode learned_log --train-length 10 --train-lengths 10 --target-position-mode fixed_start --target-token-count 1 --non-target-token-count 1 --non-target-sampling uniform --train-examples 2000 --val-examples 500 --test-examples 50 --eval-chunk-examples 50 --eval-sampling-mode random --eval-lengths 10 100 1000 10000 100000 1000000 10000000 --batch-size 64 --eval-batch-size 16 --epochs 200 --learning-rate 0.003 --weight-decay 0 --d-head 2 --alpha-log-scale-init -5
```

For another table condition, change `--alpha-mode`, `--epochs`, seed and unique output name according to the table. Leave `--max-train-steps` omitted, as in the original saved configs. Seeds 0–4 and all eight budgets would require 40 training runs; none were launched. Reducing chunk or eval batch size for memory is a modified evaluation configuration and should be recorded explicitly.

For checkpoint analysis, avoid the analyzer's default output files inside an old run. Use absolute new output paths:

```powershell
$analysisOutput = Join-Path (Get-Location) "runs/training_length/report_reproduction/analysis_new"
if (Test-Path $analysisOutput) { throw "Choose a new analysis directory." }
New-Item -ItemType Directory -Path $analysisOutput | Out-Null
& ../.venv/Scripts/python.exe -m analyze_stage3_mechanism --run-dir runs/stage3_seeds/learned_log_e200_s0 --output-json (Join-Path $analysisOutput "mechanism.json") --output-csv (Join-Path $analysisOutput "mechanism.csv")
```

This last CLI example was not executed. The audit used its read-only `load_model` and `analyze_model` functions. Analyzer limitations: it chooses target id 0 and the first non-target query/key pair, and `read_metric_delta` prefers evaluation length 10 instead of checkpoint training length. This is adequate for the report baseline; it is not a general worst-case analysis for all vocabulary types or a complete follow-up diagnostics recorder.

## Remaining limits

Original historical code/environment and shell transcripts need external provenance if exact retraining is required. Old `stage3base` checkpoints cannot be inspected because they are absent. The local ignored artifacts and validation scripts need an explicit sharing/archive decision before another machine can rely on them. Neither passing smoke tests nor loading stored checkpoints establishes successful full reproduction. Follow-up seed count, budget, evaluation lengths and intermediate diagnostic cadence remain open decisions in [PLAN.md](PLAN.md).
