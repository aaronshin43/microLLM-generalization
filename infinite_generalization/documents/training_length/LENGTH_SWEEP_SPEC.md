# Baseline training-length sweep execution specification

Execution date: October 24, 2026. The user authorized the
[length-sweep plan](LENGTH_SWEEP_PLAN.md). Preserve its descriptive hypotheses,
seeds, lengths, loss targets, success definition and nine-new-run scope.

## Frozen scope and execution

Reuse seeds 3, 4 and 6 at lengths 10 and 100 from
`runs/training_length/seed_replication_20260929_01/`; audit them against the
verified multiplier-1 branches in `coefficient_lr_branches_20261010_01` before
new training. Save exact configuration, initial hash, final weights, diagnostic
and history comparisons, source hashes and optimizer-record availability in
`reference_inventory.json`. Preserve all historical files.

Run lengths 30, 300 and 1000, in that order, with seeds 3, 4 and 6 within each
length. Run all nine from initialization through 6400 updates, with the original
baseline settings. Same-seed initial hashes must match reused baselines.
The inherited wall-time ceiling is 60 minutes, including audit, disposable
checks, scientific training and explicit evaluations. Check the ceiling before
each run and at fixed 200-update trace boundaries; retain incomplete status if
it is reached. Analysis and document generation are separate from training.

The resolved environment and optimizer settings are saved before scientific
training in the manifest. Use the repository-root `.venv`, CUDA, float32 and
the original `set_reproducibility` settings. Preserve AdamW's single parameter
group, learning rate 0.003, weight decay 0, betas 0.9/0.999 and epsilon 1e-8.
Do not change task generation, loader seeds, sampling, epochs or batch ordering.

Common length-100 BCE targets remain 0.02, **0.01 primary**, and 0.005.
Use the first adjacent-200-update downward crossing over the full recorded
trajectory from initialization; do not restrict it to post-1600. Preserve
crossing brackets and censoring. A cell is eligible for the fitted-solution
comparison only when own-length positive and negative logits at that crossing
are strictly positive and negative respectively. Retain all 15 cells in the
coverage table, including unfitted, unreached and non-finite cases.

## Recording and validity

Extend `TrainingRecorder` with an optional diagnostic grid and `finish_run`
with an optional explicit-evaluation grid. Existing callers retain their
original defaults. The sweep uses 10, 30, 100, 300, 1000 and 10000 at initialization
and every 200 updates. The training loop and optimizer are unchanged.

Before scientific training, compare a completely unobserved 64-update path
with an observed one for seed 3 at each new length. Require exact weights,
all optimizer states, RNG, train/validation loader generator states, epoch
history and update counts, including validation after both epochs. These six
disposable paths total 384 updates and are not scientific cells or replays
of the old lengths.

The compact coefficient observer records every update 1–6400, including raw
gradient, actual parameter/coefficient/score/exponent changes, before/after
Adam step and moments, bias-corrected moments, learning rate, applied normalized
direction and exact signed exponent accounting. Flush every 200 updates; store
fixed-block sums and rolling realized-batch hashes. The second moment is an
uncentered average of squared gradients. Record full model/AdamW/data/RNG/history
states at 1600 and 6400 after epoch validation. No optimizer manipulation occurs.

Reconstruct second moments at 1600 from updates 1–200 and 201–1600, and at 6400
from those intervals plus 1601–3200 and 3201–6400. Use actual saved betas and
compare the float64 weighted-history reconstruction against float32 recorded
moments with relative tolerance 0.0001 and absolute tolerance 1e-12. Preserve
the measured residuals. Do not reconstruct old missing histories.

Directly evaluate final new models on the six-length grid using the existing
50-example, 50-example chunk and batch-16 settings. Preserve historical direct
evaluation at shared lengths for old baselines; evaluate their saved final
weights only at missing lengths 30 and 300. Record direct/analytical provenance.
Class predictions, logits and masses are checked against analytical diagnostics;
predictions beyond 10000 remain analytical only.

New scientific compute is 57600 updates, 3686400 sequence exposures and
1634304000 token exposures. Each fixed dataset is revisited over 200 epochs.
Disposable checks, validation and evaluation are reported separately. All
fifteen cells, failed fits and non-finite states remain visible; no seed is
replaced, no training budget extended and no extra baseline replay is added.

## Commands and deliverable

From the repository root:

```powershell
.\.venv\Scripts\python.exe infinite_generalization/src/run_training_length_sweep.py --output-dir runs/training_length/length_sweep_20261024_01 --device cuda --wall-time-minutes 60
.\.venv\Scripts\python.exe infinite_generalization/src/analyze_training_length_sweep.py --run-dir runs/training_length/length_sweep_20261024_01
```

Write `LENGTH_SWEEP_RESULTS.md` with all seeds, equal-update and common-loss
tables, trajectory/length-response plots, optimizer availability/reconstruction,
exposure and provenance checks. Review the predeclared decreasing-coefficient,
decreasing-exponent and increasing-margin expectations separately. Select one
justified next action; do not execute a causal intervention or new experiment
family as part of this sweep.

## Completed execution

All nine new runs and six reused comparison cells completed in 7.82 minutes,
inside the 60-minute ceiling. See [LENGTH_SWEEP_RESULTS.md](LENGTH_SWEEP_RESULTS.md).
The run keeps the original pre-launch specification and plan in `source_snapshot/`;
this completion note does not alter those launch artifacts.
