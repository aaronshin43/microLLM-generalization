# Initial-Behavior-Matched Coefficient Control

Planning date: 2026-10-03. This document preserves the pre-launch design;
execution on 2026-10-04 is recorded in the [specification](MATCHED_INITIAL_BEHAVIOR_SPEC.md)
and [results](MATCHED_INITIAL_BEHAVIOR_RESULTS.md). It follows the [coefficient-initialization results](MECHANISM_FOLLOWUP_RESULTS.md),
where a larger initial coefficient changed initial attention, altered parameter
selection, and caused seed 5 to fail at initial coefficient 1.0.

## Question and fixed experiment

Does coefficient initialization still change fitting, realized updates and
selected parameters when the initial attention, logits and loss are matched at
a declared reference length? Use the same two-token Stage 3 detector, learned-log,
training conditions `[10]` and `[100]`, paired seeds 3–7, 6400 updates,
and intended initial coefficients baseline, 0.1 and 1.0. The baseline coefficient
is the actual float32 value of $\operatorname{softplus}(-5)$, approximately
0.006715. Keep the prior datasets, batch order, balanced class allocation,
2048 train/512 validation examples per length, batch 64, AdamW learning rate
0.003, zero weight decay and explicit evaluation lengths 10, 100, 1000, 10000.

The primary estimand is the **within-training-length effect of changing initial
coefficient under a specified behavior-matching transform**. Length-10 versus
length-100 contrasts remain useful, but must state their reference and pairing.
Keep fitted outcomes and failed fits visible for every seed. Seed 5 is a
predeclared case study within the full five-seed comparison, not the only run.

## Matching rule

For each seed, reconstruct its original baseline query/key/head initialization.
Let $c_b$ and $\Delta_b$ denote its baseline coefficient and raw score gap.
For reference length $n_*$, preserve the seed's signed baseline effective margin:

```math
M_b(n_*)=[1+c_b\log(1+n_*)]\Delta_b.
```

After setting the new unconstrained scalar with the softplus inverse, read its
actual float32 coefficient $c_0$. Define a positive scale:

```math
r=\frac{1+c_b\log(1+n_*)}{1+c_0\log(1+n_*)},
\qquad \Delta_0=r\Delta_b.
```

The primary transform scales both complete projection matrices symmetrically:

```math
W_{q,0}=\sqrt r\,W_{q,b},\qquad
W_{k,0}=\sqrt r\,W_{k,b}.
```

Since the projections have no bias, this scales every raw attention score by
$r$. At $n_*$, the corrected scores, attention distribution, target mass,
value output and binary logits match the baseline up to float32 tolerance.
The head stays bitwise unchanged, so positive and negative loss also match for
valid task inputs of that length. Negatives have no target and retain the same
value output and logit at every length. Do not replace a negative baseline gap
with a positive one or use a common gap across seeds: preserve each seed's
starting behavior. For the baseline coefficient, bypass scaling with $r=1$
so that its initial state remains bitwise identical.

This ratio does not divide by $\Delta_b$, and remains well-defined for a zero
or very small baseline gap. Save requested and actual coefficients, $r$, both
initial state snapshots and the measured matching residuals before training.

## What is matched and what changes

| Quantity | Relation to the paired baseline |
|---|---|
| Corrected attention scores and distribution at $n_*$ | Matched within declared numerical tolerance |
| Positive/negative logits and loss at $n_*$ | Matched, with the same head and fixed values |
| Head weights/bias, data, batch order and optimizer hyperparameters | Unchanged; head state is bitwise identical initially |
| Query/key directions and relative norm balance | Preserved by symmetric positive scaling |
| Query/key norms, raw scores and raw $\Delta$ | Changed by the specified transform; raw gap scales by $r$ |
| Initial coefficient and its unconstrained scalar | Changed |
| Initial $c\Delta$ and length dependence | Generally changed |
| Attention/logits at other positive sequence lengths | Generally unmatched; measure and report |
| Gradients, Jacobians and realized AdamW updates | Unmatched; these are central diagnostics |

Matching at both lengths 10 and 100 would generally force the original
coefficient and raw gap when $\Delta_b\ne0$. One reference therefore cannot
establish equal positive behavior over the entire length range. In particular,
a length-100 match does not establish equal starting training behavior for a
length-10 run.

Even at the matched reference, the derivatives differ. For example, with
$u_0$ the unconstrained coefficient,
$\partial M(n_*)/\partial u_0=\log(1+n_*)\Delta_0\operatorname{sigmoid}(u_0)$.
Changing the coefficient and scaling query/key changes this sensitivity and the
projection sensitivities. Identical initial loss need not produce identical
first updates under AdamW. Residual differences cannot be attributed uniquely
to the scalar, query/key scaling or the optimizer from this comparison alone.

## Main run matrix: 30 new runs

| Phase | Reference length | Training conditions | Coefficients | Seeds | New runs |
|---|---:|---|---|---|---:|
| A: common reference | 100 | `[10]`, `[100]` | 0.1, 1.0 | 3–7 | 20 |
| B: match length-10 training behavior | 10 | `[10]` | 0.1, 1.0 | 3–7 | 10 |

Reuse the ten existing baseline single-length runs only after verifying the
baseline initialization and training path with the new runner. Compare with
the twenty existing **unmatched** coefficient-intervention runs as well.
Do not resume any model-only checkpoint; all new matched runs start from the
specified initial state with a new optimizer.

Phase A gives a common reference and, for changed coefficients, identical
initial weights across the two training lengths within each seed/coefficient
pair. Its `[100]` comparison matches initial training behavior directly; its
`[10]` comparison tests transfer of the reference control and must retain the
initial length-10 mismatch. Phase B makes the `[10]` within-length comparison
match its own initial training behavior. Combining phase-B `[10]` with phase-A
`[100]` matches each condition's own training behavior to its own baseline, but
uses different projection scales across lengths. Label that contrast accordingly.

The reference-100 prediction is that the high-coefficient seed-5 failure may
weaken or disappear when its initial length-100 attention is restored. The
reference-10 companion tests the same prediction at length 10. Failure recovery,
continued failure, and differences in recovery time are all reportable. Do not
declare initial attention the sole cause if recovery occurs: query/key norms
also changed. If fitted trajectories and final exponent contrasts remain,
that is evidence that the matched reference behavior is insufficient to fix
the learning path. If they shrink, quantify how much rather than declaring
all initialization effects removed.

## Conditional check of query/key factorization

If the main runs retain coefficient-dependent fitting, update trajectories or
final parameter differences, add a predeclared sensitivity check for seeds 3–5,
initial coefficient 1.0, and each single length matched at its own training
length. Compare the main symmetric transform with:

```math
\text{query-only: }W_{q,0}=rW_{q,b},\quad W_{k,0}=W_{k,b};
\qquad
\text{key-only: }W_{q,0}=W_{q,b},\quad W_{k,0}=rW_{k,b}.
```

This adds 12 runs; reuse the six corresponding symmetric runs. At fixed seed,
coefficient and reference, all three factorizations have the same initial raw
scores and behavior **at every length**, while their projection norms and
parameter derivatives differ. Verify that stronger equality before training.
Different subsequent updates would identify sensitivity to the factorization
of the same initial scoring function. Agreement would establish robustness over
these three transforms only. This check should be decided from the predeclared
trigger, with all three seeds retained, rather than selecting a winning transform.

## Diagnostics, validity and execution order

1. Add an explicit initial-state transform before optimizer construction and
   before recording update 0. Do not repurpose the passive diagnostic observer
   to silently mutate the model. The existing default initialization path must
   remain unchanged. Save baseline/post-transform weights, source, device and
   exact transform parameters in new output directories.
2. Check the scaling identity in float64 analysis of saved float32 weights and
   run direct positive/negative sequence forwards at the reference length.
   Use a declared tolerance, initially `rtol=3e-5`, `atol=3e-6`, for corrected
   scores, attention, logits and loss, and store maximum residuals. Check head
   identity, unchanged signs/directions, unchanged data RNG state, and the
   other-length differences. Stop before training if matching fails. Do not
   reject a valid match because its original head or raw margin was negative.
3. Verify the bypassed baseline against an existing fresh-run prefix at 1600
   updates. Test that tracing preserves a short run's weights/history/RNG.
   Measure a representative 6400-step run, then execute phases A and B. Use the
   prior 60-minute ceiling as a provisional total-training limit; the 30 runs
   are roughly 12–13 minutes at the previous rate before extra logging. Decide
   the conditional factorization phase at the main-phase review.
4. Retain 200-update diagnostics and record actual gradients and updates for
   every step 1–200, plus windows 1590–1610 and 3190–3210 in all new runs.
   Record coefficient, raw gap, exponent, query/key norms and relative updates,
   head state, both-length target masses/logits/losses, and batch exposure.
   Reuse the exact midpoint decomposition of exponent and effective-margin
   updates. A first-step comparison at the matched training length is especially
   informative because the same initial loss signal enters different Jacobians.
5. Report seed-level fitting/failure and head/margin sign transitions first;
   then compare parameter trajectories at equal updates and at available matched
   **common length-100** losses. Also report loss at each run's matched reference;
   do not mix different reference losses into one matched-progress comparison.
   Record sampled threshold entry, recrossing and censoring at 6400, separating
   failed heads from fitted exponent comparisons. Compare finite failure lengths
   only under their head-validity checks and label them analytical predictions.

No new task, mixture, length-1000 training or explicit 10M evaluation is included.
The report should state the matching reference and transform beside every
comparison, including null results and the conditional factorization outcome.
