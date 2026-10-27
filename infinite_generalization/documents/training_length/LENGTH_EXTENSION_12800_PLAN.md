# Training-length continuation from 6400 to 12800 updates

Planning date: October 24, 2026. Following the
[length-sweep results](LENGTH_SWEEP_RESULTS.md), continue the existing baseline
paths at training lengths 10, 30 and 1000 with paired seeds 3, 4 and 6.
This document plans nine continuations; no training or implementation was
performed to prepare it. Existing checkpoint schemas were inspected read-only.

## Research purpose

At 6400 updates, all selected models fit their own training problem, but only
length 10 has $r=c\Delta>1$. Length 30 has $r$ between 0.817 and 0.923;
length 1000 has $r$ between 0.628 and 0.727. Their exponents are still growing.

The extension asks two distinct questions:

1. Do the length-30 and length-1000 baseline paths reach $r>1$ within a larger,
   fixed optimization budget while retaining successful fitting?
2. At equal updates, and where reachable at a common evaluation loss, does
   training length still select different coefficients, margins and exponents?

Length 10 is continued as the equal-budget reference, length 30 tests a condition
near the threshold, and length 1000 tests the longest training condition in the
sweep. Crossing the threshold and eliminating the length-dependent exponent gap
are different outcomes. Interpret both alongside the classifier head.

## Fixed scope

| Setting | Specification |
|---|---|
| Training lengths | Single lengths 10, 30, 1000 |
| Paired seeds | 3, 4, 6; nine paths, no seed replacement |
| Start | Existing state after update 6400 and epoch-200 validation |
| New training | Updates 6401–12800; 6400 additional updates per path |
| End | Total update 12800, corresponding to completed epoch 400 |
| Scaling/model | Unchanged baseline learned-log Stage 3 binary detector |
| Dataset/batches | Existing fixed datasets, 2048 balanced training examples, 512 balanced validation examples, batch size 64 |
| Optimizer | Existing AdamW state; every parameter learning rate 0.003, weight decay 0, betas 0.9/0.999, epsilon 1e-8 |
| Numerics | Preserve source CUDA, float32 and deterministic execution settings |
| Trajectory diagnostics | Existing global 200-update grid; summarize 6400, 9600 and 12800 |
| Coefficient trace | Every actual update 6401–12800 |
| Full state exports | 9600 and 12800, after epoch validation and history recording |
| Final explicit evaluation | Lengths 10, 30, 100, 300, 1000, 10000; existing balanced 50-example, chunk-50 and batch-16 settings |

Only the stopping budget and necessary output/recording metadata change. Do not
restart from initialization, reset moments, change learning rates, manipulate
optimizer state, add a scheduler, or stop a successful path when it first crosses 1.
No length below 10, length-100/300 continuation, curriculum, model extension or
explicit 10M evaluation is included. The endpoint remains 12800 even if a
condition has not crossed; a further extension is a separate research decision.

## Source checkpoints and restoration

All paths below are relative to `infinite_generalization/`. Substitute each
specified seed for `{seed}`. Historical files remain read-only.

| Length | Source files at update 6400 |
|---:|---|
| 10 | `runs/training_length/coefficient_lr_branches_20261010_01/branches/n10_s{seed}_control/`: `final_continuation_state.pt`, `model.pt`, `config.json`, `train_history.csv` and `training_diagnostics.csv` |
| 30 | `runs/training_length/length_sweep_20261024_01/new_runs/n30_s{seed}/state_step6400.pt` |
| 1000 | `runs/training_length/length_sweep_20261024_01/new_runs/n1000_s{seed}/state_step6400.pt` |

### Two existing checkpoint formats

For lengths 30 and 1000, the single bundle contains model/configuration,
optimizer and named moments, parameter order, epoch/update boundary,
history/diagnostics/exposure, loader states and Python/NumPy/Torch RNG states.
The inspected bundles record update 6400, completed epoch 200, next epoch 201,
batch cursor zero and completed validation. They have one optimizer group.

Length 10 uses the **multiplier-1 control**, whose equivalence to the historical
baseline was verified in the [branch experiment](COEFFICIENT_LR_BRANCH_RESULTS.md).
Its `final_continuation_state.pt` contains optimizer, named moments, RNG and
loader states, but does not contain model weights, configuration, history or
epoch counters. Assemble a continuation bundle using the matching files listed
above and the documented capture boundary. Verify the 6400-step weights against
the historical baseline and the final history row against epoch 200/update 6400.
Use source parent parameter metadata if needed to validate named mappings;
do not supply missing optimizer state from a model-only file.

The length-10 control optimizer has **two groups**, with the coefficient separate
but both rates equal to 0.003. Lengths 30/1000 have one group. Preserve each
source layout, parameter mapping, moments, counters and execution options.
The current `restore_parent` helper in `run_coefficient_lr_branches.py` builds
a single-group optimizer, so it cannot load the length-10 two-group state
unchanged. Adapt restoration explicitly; do not merge groups or call the
coefficient-splitting intervention as part of this continuation.

### Restoration and validity gates

Before scientific continuations:

1. Inventory and hash every source file; retain baseline verification and
   configuration provenance. Verify all nine start states have the expected
   model, coefficient/margin/exponent, head and own/common loss, successful
   fitting, and named optimizer counters at 6400.
2. Recreate the original datasets/loaders and verify tensor identities. Restore
   model and optimizer by a checked parameter mapping, then loader generators
   and RNG. Require exact restored weights, named moments, counters, group
   options and RNG/loader state against the source. Keep model mode consistent
   with the captured post-validation boundary.
3. Begin epoch 201 with the original epoch-specific batch-shuffle rule. Do not
   repeat epoch-200 validation, replay the first 200 epochs, advance restored
   generators again, or regenerate a different dataset. Restore the parent
   again after any disposable probe before scientific training begins.
4. Check save/load continuation at seed 3 for each length: compare a 64-update
   uninterrupted continuation with a 32-update continuation, boundary save/load,
   and another 32 updates. Require exact final weights, optimizer, batches,
   RNG/loaders and epoch history through update 6464. These six disposable paths
   add 384 updates, reported separately from scientific compute. Check observer
   passivity if the recording hooks are changed.
5. Save the normalized starting bundle and its provenance in the new output
   directory. This preserves the original scientific state; it does not
   normalize parameter values or optimizer groups. A missing or incompatible
   state is a restoration issue, not permission to silently train a replacement.

No uninterrupted 0–12800 reference run exists. These checks validate the
restoration and serialization procedure; do not claim a bitwise comparison
against an unavailable historical long-budget run.

## Recording and comparisons

Use $c=\operatorname{softplus}(u)$ and $r=c\Delta$ as before. Preserve the
6400 boundary once and retain prefix provenance. The continuation observer and
exposure counters start at global update 6400, not zero. Initialize fresh CSV
headers correctly when appending global-step records to a new output directory.

### Primary: threshold entry and equal-update solution selection

- Record $c$, $\Delta$, $r$, head slope/intercept/threshold, effective margin,
  target mass, positive/negative logits and balanced BCE every 200 updates on
  the six-length evaluation grid. Preserve measured epoch train/validation history.
- Record the full coefficient gradient, actual $u$/$c$/$r$ updates, before/after
  Adam moments and counters, bias-corrected moments, normalized direction and
  coefficient/score accounting terms on every new update. Keep definitions
  consistent with the sweep and report numerical residuals.
- For lengths 30/1000, record first new $r>1$ entry and any recrossing from the
  complete step trace, using a consistent numerical definition of $r$. For
  length 10, label it already above 1 at extension start and retain its earlier
  grid-bracketed entry; do not count 6401 as a newly learned threshold crossing.
  If no entry occurs, report right censoring at 12800. Record threshold retention
  separately from first entry.
- At 6400, 9600 and **12800 primary**, show all nine cells and paired length
  gaps $r_{10}-r_{30}$ and $r_{30}-r_{1000}$, plus changes since 6400. Plot the
  combined trajectories and seed-level values, not only means.
- Compare average exponent growth per update in fixed intervals 4800–6400,
  8000–9600 and 11200–12800. This describes acceleration or slowing; do not use
  a short linear extrapolation as a guaranteed threshold-crossing forecast.
- Define fitted states by strictly positive own-length positive logit and
  strictly negative own-length negative logit. Report any loss of fitting on
  the diagnostic grid and final direct class accuracies without dropping cells.

Tentative expectations are retained fitting and the ordering
$r_{10}>r_{30}>r_{1000}$ at 12800. Threshold entry for lengths 30/1000 is an open
question, not a promised result. Do not assume the ordering holds at every
intermediate update or predeclare that all runs must cross.

### Secondary: new common-loss comparisons

Use the existing balanced **length-100 evaluation BCE**. At 6400 it is about
0.0013–0.0017 in these parents. Freeze new targets **0.001 primary**, 0.0005
and 0.0001 secondary, all below the starting losses. The earlier sweep targets
0.02, 0.01 and 0.005 were reached before this extension and are historical
context, not new continuation comparisons.

Use the first post-6400 downward crossing bracketed by adjacent 200-update
records. Preserve bounds, interpolate using the established convention and
report coverage out of nine; targets not reached by 12800 remain censored.
Restrict the fitted-solution comparison to cells with strictly correct own-length
logits at the crossing, while retaining every cell and its reason in the table.
Compare coefficient, margin, exponent and head. Matching one loss does not
equalize the full function, geometry or optimizer state.

### Optimizer memory during continued training

For every path, reconstruct the second moment from its saved boundary and the
complete new gradient trace:

```math
v_T=\beta_2^{T-6400}v_{6400}
 +(1-\beta_2)\sum_{j=6401}^{T}\beta_2^{T-j}g_{u,j}^{2}.
```

At 9600 and 12800, report the surviving pre-6400 contribution, the new-gradient
contribution and their fractions of the recorded second moment. Use the sweep's
reconstruction tolerances: relative 0.0001, absolute 1e-12, retaining residuals.
For lengths 30/1000, the complete earlier histories permit an additional split
of the carryover into updates 1–1600 and 1601–6400. Keep length 10's missing
earlier gradient records explicit; its boundary carryover remains identifiable.

Relate memory fractions to actual coefficient/exponent growth. A high surviving
fraction is not a causal attribution, and a changed fraction alone does not
establish why the selected solution changed. This extension does not manipulate
moments and cannot replace the proposed causal intervention.

At 12800, verify explicit logits/masses against the reduced-model formulas and
report head-aware analytical failure status. Predictions beyond length 10000
remain analytical. A larger exponent does not by itself rank finite classification.

## Compute, outputs and execution sequence

Each path adds 6400 updates, 200 epochs and 409600 sequence exposures, balanced
by class. Its cumulative path has 12800 updates and 819200 sequence exposures.
Reuse the historical prefix without charging it as new training.

| Length | Continuations | New sequence exposures per path | New token exposures per path |
|---:|---:|---:|---:|
| 10 | 3 | 409600 | 4096000 |
| 30 | 3 | 409600 | 12288000 |
| 1000 | 3 | 409600 | 409600000 |

New scientific compute totals **57600 updates, 3686400 sequence exposures and
1277952000 token exposures**. Report disposable checks, validation/evaluation
and any implementation checks separately. Equal updates equalize sequence
exposure, but token exposure and runtime remain length-dependent.

Execution order is source audit, continuation implementation/short validity
checks, frozen manifest/specification, then nine scientific tails. Run lengths
10, 30 and 1000 in that order, seeds 3, 4 and 6 within each length. Keep source
files and existing user changes intact; use the repository-root `.venv` and
the [README setup](../../README.md).

Use a fresh ignored `runs/training_length/` directory. Before launch, record
source hashes/snapshots, state mapping, settings, probe accounting, run list,
predictions and a provisional 60-minute wall-time ceiling including checks and
evaluation. Preserve incomplete status if that ceiling is reached rather than
silently changing the scientific endpoint. Save complete 9600/12800 bundles
after validation, plus per-run status, exposure, traces and data verification.

Write `LENGTH_EXTENSION_12800_RESULTS.md` with all nine paths, crossing/censoring
and fitting outcomes, fixed-update/common-loss comparisons, head-aware
extrapolation, moment carryover, provenance and compute accounting. No command
is asserted for a continuation interface that has not yet been implemented.

## Direction review after 12800

- If longer-length paths cross while exponent gaps remain, describe budget-
  dependent threshold attainment together with persistent solution differences.
- If gaps shrink substantially, investigate how further optimization changes
  the length contrast, using growth, common loss, head and moment records together.
- If a path does not cross, conclude only non-entry within 12800 updates;
  neither permanent failure nor convergence follows from this endpoint.
- If fitting regresses or trajectories become non-finite, retain the outcome
  and separate it from exponent selection among successfully fitted solutions.

Review these results before selecting a limited second-moment intervention.
Do not automatically extend until all paths cross, add short-length conditions,
or expand another experiment family. Completing this plan does not establish
a universal training-length law or the optimizer history's causal role.
