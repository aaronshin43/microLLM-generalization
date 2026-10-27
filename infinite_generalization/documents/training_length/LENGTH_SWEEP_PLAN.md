# Baseline training-length sweep before optimizer-state intervention

Planning date: October 21, 2026. This plan adds a small descriptive length sweep
before second-moment intervention. It specifies the next experiment; no sweep
training, optimizer-state intervention, or implementation was performed to write it.

## Purpose and questions

The existing [independent-seed results](COEFFICIENT_LR_NEWSEEDS_RESULTS.md) explain
the immediate length-10/100 coefficient response using softplus sensitivity and
Adam normalization. Before developing that pair-specific explanation further,
check how parameter selection varies over a wider range of training lengths.

The central question remains: among models that successfully fit the task, how
does training length affect the selected length-generalization solution?
This sweep asks:

- Does the smaller exponent at length 100 extend smoothly to intermediate and
  longer lengths, or show a plateau, reversal, or change of regime?
- How do coefficient, raw margin, fitting progress, and head state change together?
- Do coefficient moments and actual updates follow a length-dependent pattern
  beyond the original pair, and when does that pattern develop?
- Which differences remain at a common evaluation loss?

This is a bounded exploratory map. Associations between moments and exponent
selection will motivate later causal tests; they do not establish mediation.

## Fixed experiment matrix

| Setting | Specification |
|---|---|
| Training lengths | Single lengths 10, 30, 100, 300, 1000 |
| Paired seeds | 3, 4, 6, matching the existing coefficient-rate branch parents |
| Scaling | Baseline `learned_log` only |
| Model/task | Existing Stage 3 binary presence detector; one target type, one non-target type, fixed one-hot values, width 2, fixed-start target, final non-target query, linear head |
| Initialization | Original model initialization; coefficient parameter $u_0=-5$, hence $c_0=\operatorname{softplus}(-5)$, approximately 0.006715 |
| Training budget | 6400 optimizer updates from initialization; no outcome-based early stopping |
| Data | Existing fixed-dataset construction: 2048 balanced training examples and 512 balanced validation examples per run |
| Batch size | 64, unchanged sampling and epoch shuffle rules |
| Optimizer | Baseline AdamW, learning rate 0.003 for every parameter, weight decay 0; preserve existing options, including betas and epsilon |
| Execution | Baseline CUDA, dtype and deterministic kernel settings; record resolved settings |
| Trajectory diagnostics | Initialization and every 200 updates, including 6400 |
| Coefficient trace | Every update 1–6400 in the nine new runs |
| Final explicit evaluation | 10, 30, 100, 300, 1000, 10000; existing chunking and evaluation settings |

There are 15 comparison cells: six reused baselines at lengths 10 and 100, plus
**nine new runs** at lengths 30, 300 and 1000. No mixture, new learning rate,
coefficient initialization change, curriculum, moment manipulation, new model,
or explicit 10M evaluation is included. This sweep precedes those decisions.

### Reuse and comparability

Use the seed-3/4/6 baseline paths from
`runs/training_length/seed_replication_20260929_01/` for lengths 10 and 100.
The multiplier-1 paths in
`runs/training_length/coefficient_lr_branches_20261010_01/` provide verified
continuations and additional optimizer records. Follow their documented
[baseline settings](COEFFICIENT_LR_BRANCH_SPEC.md), not a faster coefficient branch.

Before new training, inventory the six references, their configurations,
initial-state hashes, final checkpoints, diagnostics and available optimizer
records. Save source paths and hashes in the sweep manifest. Verify that the
new run configurations differ only in training length and necessary output or
recording settings. Same-seed initial model states must match across all five
lengths; preserve the existing data-generation and loader-seed rules.

Old paths have sparse optimizer traces, not complete coefficient histories.
Report availability by seed, length and step. Use shared observations such as
the saved 1600 state, existing detailed windows and endpoint states where
available. Do not infer missing moments from model weights, interpolate sparse
gradients into a complete history, or silently add six baseline replays.
If reuse is incompatible or missing, record that limitation before changing scope.

## Measurements

Use the same definitions as the earlier experiments:

```math
c=\operatorname{softplus}(u),\qquad r=c\Delta,\qquad
\alpha(n)=1+c\log(1+n),
\qquad p(n)=\sigma\!\left(\alpha(n)\Delta-\log(n-1)\right).
```

At initialization and every 200 updates, record $c$, $\Delta$, $r$, effective
margin, head slope/intercept/threshold, target mass, positive/negative logits,
and balanced BCE. Compute closed-form diagnostics on the common evaluation
grid above, which includes every own training length. The current recorder's
fixed grid omits 30 and 300; extend recording explicitly without changing training.
Keep measured epoch training/validation history alongside these analytical rows.

Define successful fitting at a recorded point by strictly positive own-length
positive logit and strictly negative own-length negative logit. Confirm the
final classifications with direct forwards. Retain all 15 cells, including
failures, ties and non-finite states. Report fitting outcomes before describing
exponent ordering among successful runs; never replace an inconvenient seed.

For each new run, the compact every-update coefficient trace should contain:

- Global update, $u$ before/after, raw gradient $g_u$, actual $\delta u$,
  $c$ before/after and actual $\delta c$.
- Coefficient Adam `step`, `exp_avg`, `exp_avg_sq`, their before/after values,
  bias-corrected moments, learning rate and applied normalized direction.
- $\Delta$ and $r$ before/after, and the coefficient/score accounting terms below.

```math
\delta r=\overline{\Delta}\,\delta c+\overline c\,\delta\Delta.
```

Bars are within-update midpoints. This exact signed split is an accounting
identity, not a causal percentage. Record residuals and signed sums in fixed
200-update blocks. Adam's second moment is an average of squared gradients,
not the centered statistical variance; distinguish raw gradients, normalized
directions and actual parameter updates in figures and interpretation.

The complete new traces also permit reconstruction of second-moment history.
At 1600, separate the surviving contributions of updates 1–200 and 201–1600.
At 6400, additionally separate 1601–3200 and 3201–6400. Check reconstruction
against recorded moments using the actual optimizer settings. This closes the
history gap for the new lengths, while the old lengths retain their documented
missing coverage. Full query/key/head update traces at every step are not required.

Save complete model/optimizer/data/RNG states at updates 1600 and 6400 for new
runs, after epoch validation and history recording, using existing boundary-state
support. Saving these states does not include a later intervention in this sweep.

## Comparisons and expectations written before the sweep

### Primary: equal updates

The primary endpoint is update 6400; show initialization and the complete
200-update trajectories, with 1600 and 3200 as fixed intermediate summaries.
Plot training length on a logarithmic axis and show every paired seed rather
than only a mean. Report adjacent-length differences in $c$, $\Delta$ and $r$,
plus individual seed values and their ranges.

The existing length-10/100 result motivates a tentative expectation of decreasing
$c$ and $r$, and increasing $\Delta$, as length increases. Test these as separate
patterns. A plateau, reversal, failed fit or disagreement across seeds is an
informative result. The two known lengths do not justify a numerical scaling law
or a requirement that moments increase monotonically over the new grid.

For moments and updates, compare common steps where records exist and summarize
fixed 200-update blocks for new runs. Examine when coefficient growth diverges
relative to fitting and head transitions. A larger second moment alone does not
predict a smaller update without considering first moment and bias correction.

### Secondary: common evaluation loss

Use balanced BCE at the fixed reference length 100, consistent with previous
comparisons. Freeze targets 0.02, **0.01 primary**, and 0.005 before launch.
Use the first downward crossing bracketed by adjacent 200-update records and
the existing interpolation convention; retain step bounds and interpolated
parameter values. Do not extrapolate beyond recorded trajectories or silently
substitute targets. Mark unreached/unbracketed targets and missing records.

Compare only the declared eligible cells at each target and report coverage
out of all 15. Common loss is descriptive and does not match head, geometry,
optimizer state or all-length behavior. Own-length loss remains a separate
progress diagnostic; equal own-length losses are different evaluation problems.

### Extrapolation and interpretation

Track sampled entry/recrossing of $r>1$, bracketing times by the diagnostic grid.
At the final explicit grid, verify both classes, masses and logits; retain
historical results on shared lengths and use saved baseline weights for any
missing 30/300 evaluation. These added evaluations require no training.

Interpret exponents with head-aware analytical failure status. A larger $r$ does
not automatically rank finite classification performance. Clearly separate direct
evaluations from predictions beyond length 10000. This fixed-budget sweep does
not establish convergence, permanence of a threshold crossing, or a universal
training-length law.

## Exposure, validity and execution order

All runs have 200 epochs of 32 updates and process 409600 sequences: 204800
positive and 204800 negative. These are repeated exposures to fixed datasets.
Equal updates and sequence exposures do not equalize token exposure or runtime.

| Training length | Sequence exposures per run | Token exposures per run | Training paths |
|---:|---:|---:|---|
| 10 | 409600 | 4096000 | 3 reused |
| 30 | 409600 | 12288000 | 3 new |
| 100 | 409600 | 40960000 | 3 reused |
| 300 | 409600 | 122880000 | 3 new |
| 1000 | 409600 | 409600000 | 3 new |

New scientific training totals 57600 updates, 3686400 sequence exposures and
1634304000 token exposures. Verify counts from realized batches, and record
validation/evaluation work and disposable checks separately. This is an
equal-update comparison; it cannot isolate a token-budget-independent effect.

Execution sequence:

1. Audit the reusable baselines. Finalize the launch manifest/specification,
   including settings, run order, source/environment hashes, missing records,
   output location and a wall-time ceiling before training. Use a fresh ignored
   `runs/training_length/` directory and preserve historical files.
2. Extend only the needed recording/runner controls. The existing pilot runner
   hard-codes length sets; do not assume its current CLI launches this matrix.
   Use the repository-root `.venv` and existing README setup. Check passive
   recording against short unobserved runs for identical weights, RNG and history,
   including the new lengths. Count these disposable checks separately.
3. Run exactly the nine new cells from initialization. Verify paired initial
   hashes, realized exposure, 6400 updates and complete coefficient traces.
   Keep recording passive and preserve baseline optimizer/kernel behavior.
4. Assemble all 15 cells, common-loss coverage, moment availability and final
   explicit/analytical evaluations. If the execution ceiling prevents completion,
   retain partial status; do not present the intended matrix as completed.
5. Write `LENGTH_SWEEP_RESULTS.md` with seed-level tables, trajectory plots,
   length-response plots, trace/provenance checks and the direction review below.

No training command is asserted here for an unimplemented sweep interface.
This planning task does not execute the experiment or add another experiment family.

## Decision after the sweep

- If the exponent pattern extends across lengths and successful seeds, refine
  a length-dependent prediction and select a limited second-moment intervention.
  Choose its tested lengths and predictions before intervention outcomes exist.
- If a plateau or reversal appears despite successful fitting, prioritize that
  change of regime when selecting a contrast; examine coefficient, margin, head
  and optimizer records together before assuming the old pair explains it.
- If fixed-update differences largely collapse at reached common losses,
  investigate optimization progress as a candidate explanation. Incomplete
  common-loss coverage cannot establish either collapse or persistence.
- If a new length fails to fit, keep the failure visible and distinguish it from
  exponent selection among successful solutions. Do not automatically rescue
  seeds, extend budgets, add lengths or expand learning-rate conditions.

The deliverable is a broader map and one justified next action. Curriculum and
model-assumption relaxation remain later candidates, following a review of this sweep.
