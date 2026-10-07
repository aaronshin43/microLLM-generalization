# Coefficient learning-rate intervention after successful fitting

Planning date: 2026-10-06. This is the next limited experiment, following the
[matched-initial-behavior results](MATCHED_INITIAL_BEHAVIOR_RESULTS.md).
Only existing measurements were reanalyzed to prepare this plan; no new model
training or optimizer updates have been run. Complete this intervention and
review the research direction before adding further experiment families.

## Question and estimand

In an already fitted model, does changing the coefficient parameter's learning
rate relative to query/key and head learning rates change the selected
generalization solution? Separate post-fitting solution selection from the
initial attention/head transition bottleneck. For this document, $r$ denotes
the exponent $c\Delta$, not the projection-rescaling ratio used in the previous
initialization control.

```math
c=\operatorname{softplus}(u),\qquad r=c\Delta,\qquad
M_L=[1+c\log(1+L)]\Delta=\Delta+r\log(1+L).
```

Training at one fixed length can improve its effective margin via either raw
gap or exponent growth. The long-length attention log-odds have leading term
$(r-1)\log n+\Delta$. Classification also depends on the head, so exponent
changes must be interpreted alongside fitting, threshold and predicted failure.

The primary estimand is the within-parent effect of changing only the learning
rate of `alpha_log_scale_unconstrained` after update 1600. It is an intervention
on $u$, not a direct learning-rate multiplier on $c$ or $r$.

## Fixed run matrix

| Setting | Specification |
|---|---|
| Model/task | Existing Stage 3 two-token binary detector, learned-log, fixed one-hot values and final non-target query |
| Training lengths | `[10]`, `[100]` |
| Initial coefficient | Original baseline `softplus(-5)`, approximately 0.006715 |
| Seeds | 3, 4, 6; retain all specified seeds |
| Common prefix | Original training path through 1600 updates |
| Branches per parent | Coefficient learning-rate multiplier $k\in\{1/3,1,3\}$ |
| Coefficient learning rates | 0.001, 0.003, 0.009 |
| Query/key and head learning rates | 0.003 in all branches |
| Optimizer | Original AdamW settings, zero weight decay, unchanged moments and step counters |
| Data and batches | Original datasets and order; identical future batches within each parent |
| End | Total update 6400, hence 4800 new updates per branch |
| Explicit evaluation | Lengths 10, 100, 1000, 10000 with existing chunked settings |
| Budget | Provisional shared 60-minute execution ceiling, including validity checks |

Six common prefixes produce 18 comparison paths. Compute each prefix once and
copy its complete state into the three continuations. The one-times branch is
the procedural and scientific control, not an optimizer restart. There are
96000 unique scheduled training updates: six times 1600 plus 18 times 4800,
equivalent to fifteen full 6400-update runs. At recent throughput that suggests
roughly nine minutes before added validation/logging overhead; measure the new
runner before finalizing a launch estimate. No new initialization, mixture,
length-1000 training, seed-5 rescue, new task, or explicit 10M run is included.

Retain 2048 balanced training examples and 512 validation examples per length,
batch size 64, fixed data generation and sampling, head architecture, and all
other baseline settings. Each complete comparison path represents 409600
balanced sequences; each continuation processes 307200 sequences. Report the
shared prefix once in actual compute accounting and per path in exposure
accounting, without multiplying the prefix's compute by three.

## Existing evidence and predictions frozen before intervention

All six existing 1600-update states meet the proposed branch gate:

| Seed | Train length | $c$ | Raw $\Delta$ | $r$ | Head slope | Positive / negative logits, own length |
|---:|---:|---:|---:|---:|---:|---|
| 3 | 10 | 0.064340 | 8.2474 | 0.5306 | 6.5572 | +3.3085 / -3.2444 |
| 3 | 100 | 0.047918 | 9.7644 | 0.4679 | 6.8372 | +3.4844 / -3.3483 |
| 4 | 10 | 0.075189 | 8.1293 | 0.6112 | 6.5130 | +3.2861 / -3.2230 |
| 4 | 100 | 0.054721 | 9.5878 | 0.5247 | 6.7900 | +3.4620 / -3.3239 |
| 6 | 10 | 0.079885 | 7.9404 | 0.6343 | 6.5533 | +3.2950 / -3.2537 |
| 6 | 100 | 0.059835 | 9.4107 | 0.5631 | 6.7382 | +3.4279 / -3.3062 |

Source: baseline `training_diagnostics.csv` in
`runs/training_length/seed_replication_20260929_01/`. Common length-100 balanced
BCE at the branch is 0.03238–0.03802, so the post-branch loss targets below are
strictly below every parent's starting loss. These are existing diagnostic
states, not newly created restart checkpoints.

Actual baseline update traces exist for selected seeds 3 and 4; seed 6 has the
200-update trajectory but no corresponding existing baseline update trace.
Do not describe seed 6 as already validated for the local update prediction.

| Seed | Length | Coefficient contribution to $r$ change, 1590–1610 | Score contribution | Signed coefficient share |
|---:|---:|---:|---:|---:|
| 3 | 10 | +0.0035395 | +0.0008239 | 81.1% |
| 3 | 100 | +0.0015539 | +0.0005390 | 74.2% |
| 4 | 10 | +0.0038845 | +0.0009028 | 81.1% |
| 4 | 100 | +0.0016302 | +0.0005896 | 73.4% |

These are signed sums of 21 realized updates, not raw gradient ratios. The
coefficient contribution is about 2.3 times greater at length 10. At
3190–3210, its signed share remains larger: 85.1–85.4% for length 10 and
76.4–77.3% for length 100. Both window contributions are positive in these
four cells. The larger length-100 raw gradient observed elsewhere is therefore
insufficient to predict a larger applied coefficient effect there.

At the very first branch update, all branches have identical weights, batch,
gradients and AdamW moments. Query/key and head updates should coincide; only
the $u$ update changes. With the observed baseline $\delta u$ and fixed
post-step gap $\Delta^+$, a local frozen-state prediction is

```math
A_L(k,1601)\approx\Delta^+\left[
\operatorname{softplus}(u+k\delta u)
-\operatorname{softplus}(u+\delta u)\right].
```

| Seed | Length | Predicted $r_{3}-r_{1}$ at update 1601 | Predicted $r_{1/3}-r_{1}$ |
|---:|---:|---:|---:|
| 3 | 10 | +0.0003328 | -0.0001109 |
| 3 | 100 | +0.0001460 | -0.0000487 |
| 4 | 10 | +0.0003731 | -0.0001243 |
| 4 | 100 | +0.0001566 | -0.0000522 |

These float64 softplus reconstructions use rounded observed $u$ increments;
they are local numerical approximations, not exact AdamW replays or 6400-step
forecasts. Their absolute effect ratio is approximately 2.28–2.38 in favor
of length 10. Compute a more exact first-step forecast from each newly captured
parent's actual gradients and moments before training its continuations; save
it before inspecting intervention results. Preserve the current directional
predictions even if a more precise forecast refines numerical rounding.

Predeclare the following testable expectations:

1. At update 1601, coefficient update allocation changes immediately, with
   $A_L(3)>0>A_L(1/3)$ and identical query/key/head updates within a parent.
2. At 6400 and primary common loss 0.01, expect
   $r_{L,3}>r_{L,1}>r_{L,1/3}$, with training classes remaining fitted.
3. Expect larger **absolute** intervention effects at length 10 for paired
   seeds 3, 4 and 6: $A_{10}(3)>A_{100}(3)$ and
   $|A_{10}(1/3)|>|A_{100}(1/3)|$. Seed 6 is a directional extrapolation
   from the other two traces, not a calibrated numerical first-step forecast.
4. Therefore expect the same-multiplier exponent gap to widen with faster
   coefficient learning and shrink with slower learning:
   $G_3>G_1>G_{1/3}$, where $G_k=r_{10,k}-r_{100,k}$.

The first expectation has a direct frozen-state basis; the later expectations
are hypotheses about feedback. Faster coefficient motion can saturate attention
and suppress subsequent gradients, while slower motion can elicit compensatory
gap/head changes. Failure of a late ordering must be reported as a failed
prediction rather than explained away by the immediate first-step identity.
Do not promise that length 100 at three-times learning rate crosses 1, that
length 10 at one-third fails, or that a cross-length gap reverses.

The read-only planning reanalysis retains
[eligibility states](../../runs/training_length/branch_lr_planning_20261006_01/branch_eligibility.csv),
[window sums](../../runs/training_length/branch_lr_planning_20261006_01/baseline_update_windows.csv),
[local forecasts](../../runs/training_length/branch_lr_planning_20261006_01/frozen_first_update_prediction.csv)
and [input hashes/method](../../runs/training_length/branch_lr_planning_20261006_01/provenance.json).

## Complete branch state and continuation validity

Existing `model.pt` files contain model weights and metadata but no AdamW or
data state. Reexecute each original baseline prefix and capture a new full
checkpoint. Model-only files remain read-only comparison references.

With 2048 examples and batch size 64, 1600 updates end training epoch 50.
Capture the boundary **after that epoch's validation and history recording**,
before materializing epoch 51's batches. This avoids reconstructing a partially
consumed batch list. The current runner materializes loader batches and then
applies a deterministic epoch-specific shuffle, so the sampler generator's
state matters in addition to the epoch number.

The saved bundle must contain:

- Model state, parameter names/order, mode and original configuration.
- AdamW parameter groups/options and every parameter's step, first moment and
  second moment, including any enabled optional optimizer state.
- Total updates 1600, completed epoch 50, next epoch 51, batch cursor zero,
  history, diagnostic and exposure counters; record that validation is complete.
- Python, NumPy, Torch CPU and all CUDA random states; each training loader's
  generator state and any validation generator state that exists.
- Training/validation dataset identities and tensor hashes, loader settings,
  batch-shuffle seed rule, and original data-generation seeds.
- Source/revision/environment, parent-state hashes, dtype/device and optimizer
  execution flags. No scheduler or mixed-precision scaler is added; explicitly
  record their absence in the original setup.

Recreate datasets/loaders deterministically, verify hashes, and then restore
the saved random/generator states. Continue at epoch 51 without replaying
epochs 1–50 or repeating epoch-50 validation. Do not both restore a consumed
generator and advance it through 50 epochs again. Future batches must be equal
between siblings within a parent; cross-length parents retain their own
original data orders rather than being forced to share a new order.

Clone the model and optimizer with a verified parameter-identity mapping.
For example, load a cloned model and a parent-layout optimizer from the saved
bundle, then explicitly move only $u$ into its learning-rate group while
retaining its existing moment tensors and step counter. Copy every other
optimizer option into that group. Separate `deepcopy` calls on a model and
optimizer without reconciling parameter references are insufficient. Branches
must own independent weight, moment and generator storage.

Change only the coefficient group's `lr` before update 1601. Do not reset
AdamW, alter bias correction, rescale moments, change query/key/head groups'
rates, or hide this intervention inside a passive observer. Group restructuring
can affect implementation details, so exact reproduction by the one-times
control is a required gate, not an assumption.

Validity checks, completed before interpreting outcomes:

1. All six new prefixes reproduce their old initialization hashes and complete
   diagnostic/history prefixes through 1600. Existing seed-3/4/6 artifacts
   lack separate 1600-step weights; compare prefix records and verify final
   one-times weights, without claiming unavailable intermediate equality.
2. At each branch, confirm both own-length classes are strictly correct,
   $h>0$ and $\Delta>0$, using direct positive/negative forwards as well as
   closed-form diagnostics. Log the common length-100 state too. If an intended
   parent fails, diagnose reproduction instead of silently substituting a seed
   or moving the branch point.
3. Sibling model states, optimizer moments/counters, datasets, generators and
   RNG states must match before changing group rates; storage must not alias.
4. A short same-rate continuation with and without group restructuring must
   reproduce weights, optimizer moments, RNG and batches exactly. A serialized
   save/load continuation must also match an uninterrupted short test. These
   validation updates are disposable probes, not additional comparison paths.
5. Verify identical gradients and query/key/head actual updates at step 1601
   across siblings. Verify coefficient moments/counter are initially identical
   and the first applied $u$ step follows its multiplier up to measured dtype
   rounding. Capture the actual groups and rates in the trace.
6. All six one-times continuations reproduce existing 6400-step baseline final
   weights bitwise, and combined prefix/tail diagnostics and history exactly.
   Check rolling hashes of every sibling's tail batches throughout 4800 updates.
   Unexpected differences invalidate the intervention until corrected; do not
   substitute a newly defined control and call it the historical baseline.

Implement an explicit optional boundary-state export and a continuation path
using the existing training semantics. Leave the default Stage 3 behavior
unchanged; run relevant Stage 3/shared-model regressions if implementation
touches shared helpers. This planning task does not implement those hooks.

Execution order: finalize the launch specification; pass short save/load and
group-restructuring checks; capture and verify all six parents; then save all
six first-step forecasts before inspecting any intervention continuation.
Only after those gates run the 18 tails, audit the six one-times controls,
and analyze the complete comparison. Keep branch order fixed in the manifest.
If validity fails, fix it before interpreting the intervention. Do not extend
the 6400 budget or add branches to repair an inconvenient scientific outcome.

## Diagnostics and analysis

Keep the existing 200-update grid, using global step numbers through 6400.
Record the branch point at 1600 once, without creating a fictitious update zero
for the continuation. Preserve prefix exposure/history separately and combine
them consistently for the baseline reproduction audit.

Trace parent updates 1590–1600 and sibling updates 1601–1800, 3190–3210,
and 6390–6400. These fixed windows capture the immediate intervention, early
feedback, a middle phase and the endpoint. Each branch has 232 traced updates;
keep all 18 branches rather than selecting those with a clean response.

At traced steps record $u$, its gradient and actual update; the requested rate,
AdamW moment/counter and normalized step; actual $c$ and $r$ changes; full
query/key gradients, weights and actual updates; norms and relative update
norms; actual classifier weight/bias updates, head slope/intercept/threshold;
and positive mass, logits and balanced loss at both lengths 10 and 100.

Use exact midpoint accounting, where $\delta$ denotes an actual step and bars
denote the before/after midpoint:

```math
\delta r=\bar\Delta\,\delta c+\bar c\,\delta\Delta,
\qquad
\delta M_L=\log(1+L)\bar\Delta\,\delta c
           +[1+\bar c\log(1+L)]\delta\Delta.
```

Split raw-gap change into query and key midpoint terms as in the existing trace.
Report signed sums and absolute magnitudes, with numerical residuals. Avoid
unstable contribution percentages when the total nearly cancels. At update
1601, query/key **parameter** updates coincide but their contribution to $r$
can differ because the midpoint coefficient has changed; do not mistake this
coefficient weighting for immediate query/key adaptation.

Primary comparisons are paired by parent and retain every seed:

| Comparison | Predeclared analysis |
|---|---|
| Equal updates | Branch-relative changes and $A_L(k,t)=r_{L,k}(t)-r_{L,1}(t)$ at 1800, 3200 and 6400; 6400 is primary |
| Equal common evaluation loss | First post-1600 downward crossings of length-100 balanced BCE 0.02, **0.01 primary**, and 0.005 |
| Equal own training loss | Secondary within-length diagnostic; never mix own-loss definitions across lengths |
| Between training lengths | $G_k(t)=r_{10,k}(t)-r_{100,k}(t)$ and $G_k-G_1=A_{10}(k)-A_{100}(k)$ at equal updates and common loss |
| Preserved fitting | Own-length positive and negative logits at every recorded point and final direct class accuracies; retain any regression |
| Extrapolation | Explicit grid accuracy, masses and logits; sampled $r>1$ entry/recrossing and head-aware analytical failure status |

Interpolate parameters only between adjacent recorded diagnostics at the first
post-branch common-loss crossing. Retain interpolation bounds and mark targets
not reached by 6400 as censored; do not extrapolate or substitute a convenient
target. A matching scalar loss does not equalize the head, geometry or the
complete optimization state. Report those quantities and changes in raw gap
alongside $r$, not just interpolated exponents.

Report the length gap already present at 1600 separately from the additional
gap formed after branching. This experiment can test a mechanism for
**post-1600 growth and selection**; it cannot causally explain the entire
initialization-to-1600 length difference. Existing parents at different lengths
also have different head, geometry and optimizer histories. All new siblings
within a parent begin identically.

Do not count exact arithmetic decomposition as a successful mechanism by
itself: it must sum to the observed exponent change by construction. The
nontrivial tests are the preregistered length-specific response, the predicted
gap movement, temporal change in allocation before later compensation, and
persistence at common loss while fitting remains successful. Separate matched-
loss trajectory alignment from the exact equal-update sum identity.

Final artifacts should include all 18 paths, six full parent states, sibling
state/data verification, one-times baseline equivalence, first-step predictions
and errors, actual-update decompositions, seed-level fitting, common-loss
comparisons, length-gap changes, explicit/analytical extrapolation labels and
compute accounting. Use a fresh ignored `runs/training_length/` directory and
write a launch specification before training, including the frozen predictions.

## Decision after this experiment: choose one next step

No later experiment is included in the 18-path scope. Evaluate the following
in order, so that failed fitting or missed predictions are not hidden by a
favorable mean exponent response:

1. If fitting regresses, or exponent selection changes but the predicted
   length-specific effect/gap does not survive, analyze the recorded head and
   query/key geometry or compensation. Treat this as a missing-factor outcome;
   do not return automatically to seed-5 rescue.
2. If fixed-update differences largely collapse at all reached common-loss
   targets and trajectories align mainly by training progress, choose an
   all-parameter learning-rate control to test a general speed explanation.
   A null effect under both comparisons is inconclusive, not affirmative
   evidence for speed.
3. If all intended fitting states are retained, update allocation predicts the
   post-1600 length contrast and intervention response, and signs persist at
   common loss across the three seeds, choose **one** held-out check: a new seed
   or an untested coefficient-rate multiplier. Freeze its length-specific
   direction and effect-size prediction before running it. Three paired seeds
   support a targeted mechanistic check, not a population-level significance
   claim.
4. Only after independent predictions repeatedly succeed, consider relaxing
   one model assumption, such as non-target-token diversity.

The immediate deliverable after the 18-path experiment is a direction review:
which predictions succeeded, which failed, how much of the observed later
length gap is accounted for, and which single next action the evidence supports.
