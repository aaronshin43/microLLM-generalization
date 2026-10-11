# Coefficient learning-rate branching execution specification

Execution date: 2026-10-10. The user authorized execution of the
[branch plan](COEFFICIENT_LR_BRANCH_PLAN.md). All scientific hypotheses, seeds,
lengths, loss targets, and the six-parent / 18-branch matrix remain as planned.
This document records implementation choices before comparison tails start.

## State and intervention

Replay seeds 3, 4, 6 at lengths 10 and 100 from the original small coefficient
initialization. At update 1600, save weights, all AdamW states and options,
Python/NumPy/Torch CPU/CUDA RNG states, training-loader generator states,
dataset hashes, completed epoch/history, exposure, and diagnostic prefix.
Capture after epoch 50 validation, immediately before epoch 51 batch generation.
There is no pending batch, scheduler, or gradient scaler at this boundary.

Restore each parent independently. Move only `alpha_log_scale_unconstrained`
into a second AdamW group, retaining its moments and step counter. Copy all
original group settings; change its learning rate to 0.001, 0.003, or 0.009.
The other parameter learning rate stays 0.003. Continue 4800 updates to 6400.
Save the final continuation state after validation and before final evaluation.

## Validity gates and forecasts

- Require the 1600-update initial hash, diagnostic rows, and epoch history to
  equal the original baseline exactly; require both own-length classes correct,
  positive head slope, and positive raw margin.
- For each parent, run two disposable 32-update continuations: original single
  optimizer group and split groups at multiplier 1. Require exact weights,
  named moments, history, batch hashes, generator states, and RNG states.
- Before any comparison tail, calculate three disposable first updates from
  the same saved parent and next batch. Save numerical forecasts separately
  from the plan's previously declared long-run predictions. These calculations
  test intervention implementation; they do not validate long-run predictions.
- Save all six forecasts, then create `predictions_frozen.json`, then start
  the 18 comparison tails.
- Require identical future batch hashes across siblings, identical first-step
  gradients/moments/query/key/head updates, and exact forecasted first-step
  exponent. Require each multiplier-1 final checkpoint, full diagnostic series,
  and epoch history to equal its original 6400-update baseline.

The scientific training budget is 96000 updates: six shared 1600-update
prefixes plus 18 continuations of 4800 updates. Disposable grouping checks add
384 updates and disposable forecasts add 18. The 60-minute execution ceiling
includes these checks and final explicit evaluation. No additional experiment
family is authorized by this specification.

## Measurements and analysis

Keep analytical diagnostics every 200 updates and the planned detailed windows:
1590–1600 in each parent; 1601–1800, 3190–3210, and 6390–6400 in each tail.
The latter contain 232 detailed records per branch. Record gradients, actual
updates, coefficient Adam moments/counter, head updates, attention mass,
balanced evaluation loss, and query/key norms using the existing trace tools.

In addition, accumulate the exact signed decomposition across **all** tail
updates, reporting compact sums every 200 updates:

```math
\delta r=\overline{\Delta}\,\delta c+\overline c\,\delta\Delta.
```

Midpoints are within the same actual update, so this equality includes the
cross term exactly. The split is an accounting identity, not a causal mediation
estimate. Hash every realized batch, including its labels and global step.
Logging must not change model weights or RNG evolution.

Use the predeclared common length-100 balanced BCE targets 0.02 and 0.01,
with 0.005 secondary. Report first post-branch downward crossing and linear
interpolation between neighboring 200-update records; label censored targets.
Compare fixed-update and common-loss exponents, training success, threshold
entry/retention, predicted analytical failure, update allocation, and the
length gap's growth after 1600. Assess every selected seed separately.

The existing plan predicts stronger absolute intervention effects at length 10,
fast/control/slow exponent ordering, and fast/control/slow ordering of the
length gap. Preserve these predictions even if the result disagrees. After
this limited experiment, select exactly one next research direction using the
plan's decision rule; do not execute another experiment family automatically.

## Command

From the repository root:

```powershell
.\.venv\Scripts\python.exe infinite_generalization/src/run_coefficient_lr_branches.py --output-dir runs/training_length/coefficient_lr_branches_20261010_01 --device cuda --wall-time-minutes 60
```

The run directory stores the source snapshot, environment/arguments, immutable
parent files, frozen forecasts, branch checkpoints, verification files, traces,
diagnostics, batch hashes, and a completion status written after all gates pass.
