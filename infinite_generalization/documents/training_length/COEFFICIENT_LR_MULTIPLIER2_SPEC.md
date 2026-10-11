# Held-out coefficient multiplier 2 execution specification

Execution date: 2026-10-10. The user authorized the multiplier-2 experiment
selected in the [previous direction review](COEFFICIENT_LR_BRANCH_RESULTS.md).
Reuse its six full 1600-update parents: seeds 3, 4, 6 and training lengths 10/100.
Restore model, AdamW moments/counters/options, epoch/data-generator and RNG state.
Keep original small-coefficient initialization and all other training settings.

## Fixed intervention and predictions

Change only the unconstrained coefficient parameter learning rate to 0.006,
twice the original 0.003. Query/key and head rates stay 0.003. Run six new tails
of 4800 updates to total update 6400. No prefix replay, optimizer restart, new
seed, mixture, task change, training-length change, or additional rate is included.
The inherited execution ceiling is 60 minutes; expected runtime is about 3 minutes.

Copy and SHA-256 hash the existing twelve forecast rows from
`coefficient_lr_branches_20261010_01/analysis/held_out_multiplier2_predictions.csv`
before training. Do not recompute the response model or change its ±20% tolerance.
The source file retains its historical unexecuted marker; this experiment records
outcomes in a separate directory instead of overwriting the forecast record.

Completion record: the valid `_02` run completed all six continuations. The first
`_01` attempt was excluded after an exact gradient gate detected missing CUDA
deterministic-kernel settings. The correction changed execution settings to the
historical baseline, without changing the frozen prediction model or tolerance.
See [results](COEFFICIENT_LR_MULTIPLIER2_RESULTS.md) for outcomes and discarded compute.

```math
g_k=r_k-r_0,\qquad
\beta=\frac{\log(g_3/g_{1/3})}{\log9},\qquad
\widehat r_2=r_0+g_1 2^\beta.
```

Evaluate all six parents at total update 6400 and at first post-branch common
length-100 balanced BCE 0.01. Numeric success requires the actual effect
$r_2-r_1$ to lie within the **stored** lower/upper bounds, ±20% of the predicted
effect. This is a declared criterion, not a confidence interval. Also test
$r_1<r_2<r_3$, larger positive effect at training length 10, and $G_1<G_2<G_3$
for each seed. Retained fitting is required for the overall predictive claim.
Keep every failed or censored prediction; do not change tolerances after results.

## Validity and recording

Require the parent-run completion status and source hashes to match the previously
validated continuation implementation. Copy all six parent files into frozen
inputs, verify parent/state hashes, and preserve the original forecasts unchanged.
Run one disposable first-step implementation check per parent before any full
tail; these six updates do not fit or modify the long-horizon forecast model.
Apply the original `set_reproducibility` process settings before restoring RNG:
deterministic algorithms enabled with `warn_only=True`. Kernel-selection flags
are process settings and are not part of the serialized RNG states.

Require first gradients and moments, first query/key/head updates, and rolling
hashes of every tail batch to match the previous multiplier-1 sibling exactly.
Check the complete prefix history and immutable parent/forecast files. Reuse the
already verified historical multiplier-1 and multiplier-3 paths as comparators.
No additional full control replay is needed when the implementation/environment
hash gates pass. Preserve independent model/optimizer storage on every restore.

Keep analytical diagnostics every 200 global updates and detailed windows
1601–1800, 3190–3210, 6390–6400: 232 trace records per tail. Record actual updates,
Adam moments/counters, head state, attention, query/key norms and compact signed
allocation for all 4800 updates. Save final model plus optimizer/data/RNG state.
Explicit evaluation uses lengths 10, 100, 1000, 10000; larger-length failure is
analytical only. Threshold entry and retention use the 200-update recording grid.

Scientific compute is 28800 new updates, or 1843200 sequences, plus six disposable
first-update checks. The existing six shared prefixes are reused without new
training. Analyze the frozen numerical forecasts, directions, fitting, common-loss
comparison, length-gap growth, compensating updates and extrapolation. Report the
scope limit: a new rate on the same parents is not independent-seed evidence.

## Command

From the repository root:

```powershell
.\.venv\Scripts\python.exe infinite_generalization/src/run_coefficient_lr_holdout.py --parent-run-dir runs/training_length/coefficient_lr_branches_20261010_01 --output-dir runs/training_length/coefficient_lr_multiplier2_20261010_02 --device cuda --wall-time-minutes 60
```
