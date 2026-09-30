# Learned-Log Parameter-Selection Follow-up Plan

## Scope and starting evidence

Keep the Stage 3 binary detector, learned-log multiplier, fixed reused datasets,
2048 balanced training examples and 512 validation examples per length, batch 64,
AdamW at learning rate 0.003 with zero weight decay, 6400 updates, and the existing
200-update diagnostics. Preserve the explicit evaluation grid 10, 100, 1000,
10000. Do not add length-1000 training, another task, or a 10M sequence run.
Start every run from its seed initialization with a new optimizer; the existing
model-only checkpoints cannot faithfully resume AdamW.

The [6400-update result](CONFIRMATION_6400_RESULTS.md) found that seeds 0–2 all
ended with `[10]` above `[10,100]` above `[100]` in $c\Delta$, with only `[10]`
observed above 1 by 6400. The mixture-minus-`[100]` exponent gap was small
(mean 0.026 at 6400), so a new-initialization check comes before causal claims.
These are the observations to test, not assumed outcomes.

## 1. Independent-seed replication

Run learned-log at 6400 updates for seeds 3–7, each paired across `[10]`, `[100]`
and `[10,100]`: 15 new runs. Use the current coefficient initialization
`alpha_log_scale_init=-5` (initial $c=\operatorname{softplus}(-5)\approx0.006715$).
Verify identical initial model states across the three length conditions for each
seed and record realized class, length, sequence and token exposure. Keep the
existing 200-update trajectories and the same head-aware analytical failure
prediction. The old seeds 0–2 remain a separately labeled discovery set; show
all eight seeds as well as the new five, without selecting successful runs.

Primary per-seed comparisons at 6400 are `[10]` versus both alternatives and
`[10,100]` minus `[100]` for $c\Delta$, with $c$ and raw $\Delta$ shown beside
them. Report sign counts, paired differences and their spread; the small
mixture advantage should not be described as replicated if its sign varies
substantially. For each run report first sampled crossing interval of strict
$c\Delta>1$, any later sampled recrossing, and right-censor runs that have not
crossed by 6400. "Stayed above" means through the last recorded update only.
Compare the per-seed head-aware failure-length ordering at 1600 and 6400, but
do not replace exponent comparisons with finite accuracy or predicted failure
length. A seed with an invalid head threshold must be flagged rather than ranked.

Decision gate: proceed to the initialization intervention regardless of whether
all signs replicate. If either ordering fails frequently, make that variability
the outcome and inspect which parameter or head changed; do not extend the seed
sweep indefinitely merely to recover the original pattern.

## 2. Coefficient-initialization intervention

Use seeds 3–7 paired across `[10]` and `[100]` with intended initial coefficients
$c_0\in\{\operatorname{softplus}(-5),0.1,1.0\}$. This is a 30-cell comparison:
reuse the 10 baseline runs from step 1 and make 20 new runs for 0.1 and 1.0.
Set the unconstrained scalar with the softplus inverse
$u_0=\log(\exp(c_0)-1)$; the two new approximate values are -2.25217 and
0.541325. Record the actual float32 $c_0$, not just the requested value.
Ensure each seed has bitwise-identical initial query, key and head weights across
all six length/initialization combinations, while only this scalar differs.
Keep each condition's dataset and batch-order rules unchanged. Record the initial
$\Delta$, head, $\alpha(10)\Delta$, $\alpha(100)\Delta$, and positive attention
mass at both reference lengths. Changing $c_0$ changes starting attention and
the optimization path; this experiment measures sensitivity to that intervention.

For each $c_0$, compare paired `[10]` minus `[100]` trajectories for $c$,
$\Delta$, $c\Delta$, common-length loss, and head-aware failure prediction.
Report threshold entry/censoring and actual exposure. Examine differences at
equal updates and approximately matched common length-100 balanced BCE, noting
when a common loss range does not exist. An unchanged ordering across all three
initializations supports robustness over these starting states. A disappearance
or reversal identifies initialization dependence. Do not interpret either result
as a general training-length law or as isolating $c$ from initial attention.

If the effect is large, specify a subsequent *separate* control matching the
initial effective margin at one declared reference length, e.g. 100, before
running it. Matching $\alpha(100)\Delta$ changes score parameters and does not
automatically match query/key parameterization, gradients, the head, or attention
at other lengths. No such control is included in this first intervention.

## 3. Gradient and realized-update trace

After seeing steps 1–2, trace predeclared paired seeds 3–5 for both single lengths
at baseline $c_0$ and 0.1: 12 fresh 6400-update replays. If 1.0 alone changes
the ordering, add its six matching traces and label that addition. Keep the same
initialization, data order and optimizer as the corresponding sweep runs; do not
start from their model-only checkpoints. Instrument the training step passively
around `backward()` and `optimizer.step()`. Record every update for steps 1–200,
then short windows around 1600, 3200 and the first observed $c\Delta=1$ crossing
where applicable. Retain the original every-200 diagnostic record. Validate
that tracing leaves weights and the ordinary diagnostic/history prefix identical
to an untraced run; measure logging overhead before doing all replays.

At each traced update store the batch length/class counts and loss; the
unconstrained coefficient $u$, its raw gradient and its *actual* AdamW update;
the derived $c$ before/after; query and key gradients, update norms and vectors
needed to reconstruct the raw score gap; raw $\Delta$ before/after; and head
slope/intercept/threshold before/after. Compute positive attention mass and
positive logit from the exact two-score formula at both lengths 10 and 100
before/after the update. Use float64 analysis of detached weights where useful,
without inserting extra gradient or RNG work into the optimizer path.

Let $\delta x=x_{\mathrm{after}}-x_{\mathrm{before}}$ denote a one-step change
and $\bar x=(x_{\mathrm{before}}+x_{\mathrm{after}})/2$. Attribute the realized
change in $c\Delta$ exactly using the symmetric product decomposition:

```math
\delta(c\Delta)=\bar\Delta\,\delta c+\bar c\,\delta\Delta.
```

For a reference length $n$ with $L=\log(1+n)$, decompose the effective margin
$M_n=(1+cL)\Delta$ in the same way:

```math
\delta M_n=(1+\bar cL)\,\delta\Delta+L\bar\Delta\,\delta c.
```

With $\Delta=q\cdot(k_{\mathrm{target}}-k_{\mathrm{non}})/\sqrt{d}$, also
split the *realized* raw-gap change into query and key contributions using the
midpoint of the other factor. This exact update-level accounting matters because
the scalar passes through softplus, $\Delta$ is bilinear in query/key weights,
and AdamW's applied step is not proportional to a raw gradient. Check that the
recorded pieces sum to the observed $\delta(c\Delta)$, $\delta M_n$ and
$\delta\Delta$ within numerical tolerance.

Compare whether length 100 initially gets more effective-margin gain from
query/key changes and whether later coefficient gains shrink relative to length
10; report the full temporal pattern rather than only mean gradient magnitudes.
Relate any observed update allocation to the initialization intervention.
The trace is mechanistic description, not causal identification by itself.

## Execution and reporting discipline

Extend the sweep runner to select length sets and initial $c$ values explicitly,
and put the per-update trace behind an opt-in hook. Test the softplus inverse,
pairing, exact decompositions, and passive-hook equivalence; compare a fresh
baseline prefix with an existing run before launching new seeds. Write a fixed
launch specification and expected/falsifying patterns before each phase; use
new output directories with code/environment snapshots and preserve the existing
results. The prior nine-run confirmation took 3.76 minutes, making
the 35 new sweep runs roughly 15 minutes at that rate; traces add time and
logging overhead, so measure one traced run and provisionally retain the earlier
60-minute wall-time ceiling for this sequence. This is a runtime
estimate, not a guarantee. Stop at phase boundaries if the cap or a verification
check fails, and report completed cells without filling gaps by extrapolation.

The final report should show all seed-level paired contrasts, threshold crossing
intervals/censoring, coefficient-initialization response curves, and update
decomposition residuals. Separate measured explicit evaluations from analytical
failure lengths and distinguish sensitivity evidence from an identified cause.
