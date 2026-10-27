# Training-length continuation execution specification

Execution date: October 24, 2026. Execute the authorized
[12800-update extension plan](LENGTH_EXTENSION_12800_PLAN.md) with nine
continuations at lengths 10/30/1000 and seeds 3/4/6. Preserve every source
scientific state and the predeclared ordering/retained-fit expectations.
Threshold entry for lengths 30/1000 remains an open question.

## Restoration and probes

Length 10 combines its multiplier-1 final model, optimizer/RNG/loader state,
configuration, exposure, 200-epoch history and diagnostics with parameter
metadata from the matching parent. Verify final model/history/diagnostics
against the historical baseline. The post-validation mode is inferred as
evaluation mode from the recorded capture boundary and the original loop.
Lengths 30/1000 use the sweep's complete `state_step6400.pt` bundles.

Save normalized starting bundles and per-file hashes under `starts/` and
`source_inventory.json`. Normalization is schema assembly only. Preserve the
two source optimizer groups at length 10 and the single group elsewhere.
Check the integer-ID-to-name mapping against every named saved moment tensor.
Construct AdamW with those exact parameter lists/options, load the full state,
and require exact model/optimizer/RNG/loader state. All parameter counters
must be 6400. No group merging, splitting or rate intervention is used.

Recreate fixed datasets with original seeds and verify tensor hashes before
restoring loader generators. Restore RNG last. Start epoch 201 without
repeating validation or advancing generators at the source boundary.

For seed 3 at each length compare an uninterrupted, diagnostically unobserved
64-update continuation with an observed 32-update continuation, serialized
post-validation boundary, and another observed 32 updates. Exact final model,
optimizer, named moments, RNG, loaders, all batches and epoch history must match.
This jointly checks serialization and recording passivity using the six planned
paths: 384 discarded updates. Only a passive batch-identity witness is present
on the unobserved path. Restore the original start anew for scientific tails.

The coefficient recorder gains a global starting-step argument, defaulting to
zero for existing callers. Fresh trace files detect the need for a header from
file existence/size rather than the global step. The scientific training loop
and AdamW options remain unchanged. Unit-test CPU paths are separate checks.

## Frozen execution and recording

Use `runs/training_length/length_extension_12800_20261024_01/`. Run lengths
10, 30, 1000 in that order, seed 3, 4, 6 within each length. Use baseline CUDA,
float32 and original deterministic kernel settings, recorded in the manifest.
The provisional 60-minute ceiling includes audit, probes, scientific training
and explicit evaluation. Check it before each path and epoch/200-update
boundary; retain incomplete status when reached. No endpoint expansion.

Only the budget (12800 updates / epoch 400), output location and recording grid
metadata change. The full 0–6400 diagnostic/history prefix is retained once.
Length-10 missing length-30/300 analytical prefix rows are derived from saved
parameters and marked in provenance; original shared rows are retained.
No earlier missing gradient history is filled.

Record diagnostics every 200 global updates on 10, 30, 100, 300, 1000, 10000;
trace every actual update 6401–12800. Save complete states at 9600 and 12800
after epoch validation/history recording. Export cumulative and new-only
exposure separately. Directly evaluate final models with 50 balanced examples,
chunk size 50 and batch size 16 at the six declared lengths.

For exact new threshold entry, use the trace definition of the raw margin:
compute the query/key dot product in detached float64 from actual float32
weights, and multiply by the actual float32 softplus coefficient. This avoids
mixing the diagnostic's float32 raw-score subtraction with an exact-step entry.
Retain grid diagnostics with their original definition; report numerical
differences and verify crossing brackets consistently from trace before/after
values. Length 10 is already above 1 at start; retain its historical grid entry.

At equal updates summarize 6400, 9600 and 12800, paired exponent gaps and their
change, plus fixed-window growth over 4800–6400, 8000–9600 and 11200–12800.
Fit requires strictly positive own-length positive logit and negative negative
logit. Retain failures and non-finite trajectories with explicit status.

Freeze post-6400 length-100 balanced BCE targets 0.001 primary, 0.0005 and
0.0001 secondary. Use first adjacent-200-update downward crossing, established
linear interpolation, strict own-length fitting eligibility and nine-cell
coverage. Do not extrapolate censored targets or reuse earlier sweep targets
as new extension comparisons.

## Second-moment reconstruction and interpretation

At 9600 and 12800 reconstruct the recorded uncentered second moment using the
saved step-6400 moment plus all new squared raw gradients, with actual beta2.
Keep the sweep tolerances: relative 0.0001, absolute 1e-12. Retain failed checks
and residuals if float32 arithmetic exceeds tolerance; do not change thresholds
after seeing results. For lengths 30/1000 additionally split the inherited
history at update 1600, preserving both its float64 reconstruction error and
the exact saved boundary carryover. Length 10 has identifiable boundary memory
without a complete earlier history; no inferred old gradients.

These are weighted history contributions and signed update accounting,
not causal mediation. Fixed parameters' head-aware analytical extrapolation
beyond 10000 remains distinct from direct evaluation. There is no unavailable
uninterrupted 0–12800 baseline to claim as a bitwise reference.

New scientific training totals 57600 updates, 3686400 sequence exposures and
1277952000 token exposures. Probes add 384 updates / 24576 training sequence
exposures / 8519680 tokens, plus 6144 validation exposures / 2129920 tokens.
Scientific validation adds 921600 sequence exposures / 319488000 tokens.
Final explicit evaluation adds 2700 examples / 1716000 tokens. Existing prefixes
are reused; automated-test CPU work is recorded separately.

## Commands and deliverable

```powershell
.\.venv\Scripts\python.exe infinite_generalization/src/run_training_length_extension.py --output-dir runs/training_length/length_extension_12800_20261024_01 --device cuda --wall-time-minutes 60
.\.venv\Scripts\python.exe infinite_generalization/src/analyze_training_length_extension.py --run-dir runs/training_length/length_extension_12800_20261024_01
```

Write `LENGTH_EXTENSION_12800_RESULTS.md` with all nine paths, threshold/censoring,
fitting, equal-update and common-loss comparisons, growth/memory/head records,
provenance and compute. Review one next direction before any further training
or second-moment intervention.

## Completed execution

All nine paths completed at total update 12800 in 6.37 minutes, inside the
60-minute ceiling. See [LENGTH_EXTENSION_12800_RESULTS.md](LENGTH_EXTENSION_12800_RESULTS.md).
The original plan and this pre-launch specification remain in the immutable
source snapshot; this completion note does not change the frozen launch artifacts.
