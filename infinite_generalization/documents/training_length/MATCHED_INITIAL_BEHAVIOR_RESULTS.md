# Initial-behavior-matched coefficient control results

Execution date: 2026-10-04. The [plan](MATCHED_INITIAL_BEHAVIOR_PLAN.md) and
[launch specification](MATCHED_INITIAL_BEHAVIOR_SPEC.md) fix the intervention,
paired seeds, diagnostics and conditional factorization check. All 30 main
controls, 12 conditional factorization controls and two baseline verification
replays completed. Their combined training/evaluation execution took 1581.7
seconds (26.4 minutes), below the shared 60-minute ceiling. All 44 runs started
fresh at their specified seed-specific initialization with 6400 AdamW updates.
Existing model-only checkpoints were comparison objects, never training
restarts. The analysis includes these 42 controls and 30 existing baseline or
unmatched intervention comparators; verification replays are audited separately.

## Findings from the main controls

**Matching the initial attention, logits and loss at the training length did
not remove coefficient sensitivity.** Seeds 3, 4, 6 and 7 fitted both single
lengths at both changed coefficients, but seed 5 retained the failed branch
at $c_0=1$. At $c_0=0.1$, its length-100 match also failed to fit by 6400,
although the previous unmatched intervention fitted by that budget.

Among fitted runs, larger coefficients still selected larger $c\Delta$ and
smaller raw $\Delta$ than the original small-coefficient baseline, including
at matched common length-100 loss. Matching one initial behavior therefore
does not specify a unique optimization path. The transform also changes
query/key norms and their parameter derivatives; these results do not isolate
the scalar as the sole cause.

**Changing query/key factorization changed the realized updates, but did not
change fitting outcomes in the conditional check.** With the same coefficient
and initially matched scoring function at every length, the three transforms
all fitted seeds 3 and 4 and all failed seed 5 by 6400. Their first margin
updates differed substantially, while final exponent differences were modest.

## What was matched

The baseline coefficient is the actual float32 `softplus(-5)`, 0.006715348456.
Changed coefficients are the actual float32 values 0.100000001490 and 1.0.
For each seed, preserve its signed baseline scores and unchanged head:

```math
r=\frac{1+c_b\log(1+n_*)}{1+c_0\log(1+n_*)},\qquad
W_q'=\sqrt{r}W_q,\quad W_k'=\sqrt{r}W_k.
```

| $c_0$ | Reference $n_*$ | Raw-score scale $r$ | Query/key norm scale |
|---:|---:|---:|---:|
| 0.1 | 100 | 0.705428 | 0.839898 |
| 0.1 | 10 | 0.819577 | 0.905305 |
| 1.0 | 100 | 0.183610 | 0.428497 |
| 1.0 | 10 | 0.299039 | 0.546844 |

All 30 main models passed direct reference-length checks of corrected scores,
attention, value outputs, logits and per-example BCE. Their largest absolute
reference residual was 2.98e-8, within the declared `rtol=3e-5`, `atol=3e-6`.
Every reconstructed baseline initialization hash matched its existing run;
head weights/bias were bitwise equal, and the transform consumed no random
state. Both initial state snapshots are saved per run.

Only positive behavior at the declared reference is matched to baseline.
Negative logits agree at every length. Other positive lengths, raw gaps,
norms, $c\Delta$, Jacobians and updates can differ. In particular, the
reference-100 length-10 runs do not start with equal length-10 training
behavior; their reference-10 companions provide that comparison.

## Final parameters and fitting: all five seeds

Each row retains all seeds 3–7 in its mean, including failed fits. A fit means
both training classes are correctly classified at 6400; the explicit grid
also contains lengths 10, 1000 and 10000. Successful main controls were correct
on all four explicit lengths. Failed controls had 0 positive accuracy and
1 negative accuracy on the entire grid.

| Initial $c$ | Train / reference | Fitted | Mean final $c\Delta$, all five | Mean final $c\Delta$, fitted subset |
|---:|---|---:|---:|---:|
| 0.006715 | 10 / baseline | 5/5 | 1.2061 | 1.2061 |
| 0.006715 | 100 / baseline | 5/5 | 0.7295 | 0.7295 |
| 0.1 | 10 / 100 | 5/5 | 2.5659 | 2.5659 |
| 0.1 | 10 / 10 | 5/5 | 2.5476 | 2.5476 |
| 0.1 | 100 / 100 | 4/5 | 1.3961 | 2.0089 |
| 1.0 | 10 / 100 | 4/5 | 2.7399 | 4.3922 |
| 1.0 | 10 / 10 | 4/5 | 2.7299 | 4.3807 |
| 1.0 | 100 / 100 | 4/5 | 1.9330 | 2.9968 |

Seed 5 is the failed member in each 4/5 row. The matched controls contain
26 fitted and four failed runs. Means over successful subsets describe those
subsets; they are not estimates over all five initializations.

Own-reference final $c\Delta$, with the same seed's initial training behavior
matched to baseline:

| Seed | $c_0=0.1$, train/ref 10 | $c_0=0.1$, train/ref 100 | $c_0=1$, train/ref 10 | $c_0=1$, train/ref 100 |
|---:|---:|---:|---:|---:|
| 3 | 2.5460 | 1.9383 | 4.3669 | 3.0013 |
| 4 | 2.6965 | 2.0155 | 4.4037 | 3.0076 |
| 5 | 2.0738 | -1.0551 (failed) | -3.8736 (failed) | -2.3223 (failed) |
| 6 | 2.7046 | 2.0528 | 4.3362 | 2.9943 |
| 7 | 2.7170 | 2.0290 | 4.4161 | 2.9838 |

For the four pairs fitted at both lengths, the own-reference mean
length-10-minus-length-100 exponent differences are +0.6571 at 0.1 and
+1.3840 at 1.0. Raw-gap differences are +0.3306 and +0.6144, respectively,
compared with -1.4906 for the five baseline pairs. The raw-gap reversal seen
in the unmatched interventions therefore survives this behavior control
among fitted pairs. This cross-length comparison uses different projection
scales for references 10 and 100. The common-reference-100 comparison uses
identical initial weights across lengths within each seed/coefficient pair
and is separately retained in the [paired data](../../runs/training_length/matched_initial_20261004_01/analysis/length_pairs.csv).

![Own-reference exponent trajectories for all seeds and common-loss trajectories for seed 5](../../runs/training_length/matched_initial_20261004_01/analysis/matched_trajectories.png)

Changing the length-10 reference from 100 to its own length changes the fitted
mean exponent only modestly here: 2.5659 to 2.5476 at 0.1 and 4.3922 to 4.3807
at 1.0. It does not recover seed 5's coefficient-1 failure.

## Threshold entry and analytical failure

First strict $c\Delta>1$ observation on the retained 200-update grid:

| Seed | Baseline 10 / 100 | Matched 0.1, own-ref 10 / 100 | Matched 1.0, own-ref 10 / 100 |
|---:|---|---|---|
| 3 | 5400 / >6400 | 400 / 400 | 400 / 400 |
| 4 | 4200 / >6400 | 400 / 400 | 400 / 600 |
| 5 | 4400 / >6400 | 3000 / >6400 (failed) | >6400 / >6400 (both failed) |
| 6 | 4200 / >6400 | 400 / 400 | 200 / 200 |
| 7 | 4200 / >6400 | 600 / 400 | 400 / 400 |

All 26 fitted main controls crossed, with no later recording at or below 1
through 6400. The four failed controls did not cross. Crossings are sampled
brackets, not exact first-entry times; persistence is observed only through
this budget. The reference-100 length-10 crossings equal the corresponding
own-reference length-10 entries in this table.

All fitted changed-coefficient main models also passed the head-aware exact
two-score check for no finite positive-class threshold crossing. Those
analytical predictions are tied at unbounded failure length; an exponent
ordering does not give a strict failure-length ranking among them. Baseline
length-100 models retain finite predicted failure lengths despite perfect
accuracy through 10000. Failed fits must be kept out of extrapolation rankings;
for example, the seed-5 coefficient-0.1 length-100 model has a positive head
but is already wrong on the explicit training length.

## Progress matching and actual first updates

At common length-100 balanced BCE 0.01, the within-length matched-minus-baseline
parameter differences remain substantial:

| Train/ref | Initial $c$ | Pairs reaching the common loss | $c\Delta$ difference range | Raw $\Delta$ difference range |
|---|---:|---:|---:|---:|
| 10/10 | 0.1 | 5/5 | +1.064 to +1.548 | -3.065 to -2.229 |
| 100/100 | 0.1 | 4/5 | +1.212 to +1.237 | -4.696 to -4.586 |
| 10/10 | 1.0 | 4/5 | +3.158 to +3.294 | -6.291 to -6.127 |
| 100/100 | 1.0 | 4/5 | +2.159 to +2.277 | -8.321 to -7.935 |

At the same common BCE, the matched length-10-minus-length-100 exponent
difference is positive in every comparable pair; the own-reference ranges
are +0.493 to +0.582 at 0.1 and +1.213 to +1.301 at 1.0. Raw-gap differences
are also positive, unlike baseline. Similar within-length differences persist
at BCE 0.005. These observations weigh against a description based only on
different step counts. They do not equate the entire optimization state:
parameters are linearly interpolated between 200-update diagnostics at the
first loss crossing, and one scalar loss does not match head, norms or all
length-dependent behavior. Unreached loss targets remain censored in the
[loss-matched table](../../runs/training_length/matched_initial_20261004_01/analysis/loss_matched.csv).

The very first update provides a direct geometry diagnostic. For seeds 3–5
at own-reference coefficient 1, relative to their baseline first update:

| Training/reference length | Scalar gradient ratio | Applied $u$ update ratio | Applied $c$ change ratio | Query gradient norm ratio | Query update norm ratio | Effective-margin update ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 28.243 | 1.000–1.002 | 94.36–94.52 | 1.829 | approximately 1 | 1.865–1.938 |
| 100 | 17.341 | 1.001–1.009 | 94.41–95.18 | 2.334 | approximately 1 | 2.389–2.457 |

Ratios of signed effective-margin updates describe their magnitude here; the
initial direction varies by seed. AdamW largely normalizes the first raw
gradient scale, while softplus and the projection Jacobians convert similar
parameter steps into different function changes. At a matched reference,
the scalar sensitivity ratio is

```math
\frac{\partial M/\partial u_0}{\partial M_b/\partial u_b}
=r\frac{\operatorname{sigmoid}(u_0)}{\operatorname{sigmoid}(u_b)},
\qquad
\frac{\|\nabla_{W_q'}L\|}{\|\nabla_{W_q}L_b\|}=\frac1{\sqrt r}.
```

The measured ratios agree with these identities. The coefficient gradient
alone would not describe the applied update. The trace records full projection
updates and relative norms, separates query/key contributions to raw-gap
change, and decomposes exponent and effective-margin change exactly.

## Seed 5: matching the initial state does not keep attention matched

At each own-reference length, seed 5 starts with the same attention and head
as baseline, preserving its negative initial gap and head slope. After 200
updates, the paths have separated:

| Train/ref | Initial $c$ | Target mass after 200, at training length | Head slope after 200 |
|---|---:|---:|---:|
| 10/10 | baseline | 0.009370 | -0.5491 |
| 10/10 | 0.1 | 0.006298 | -0.5898 |
| 10/10 | 1.0 | 0.001962 | -0.7058 |
| 100/100 | baseline | 0.000833 | -0.5546 |
| 100/100 | 0.1 | 0.000439 | -0.6184 |
| 100/100 | 1.0 | 0.000105 | -0.7516 |

The coefficient-1 mass is approximately five times smaller than baseline
at length 10 and eight times smaller at length 100. Thus restoration of the
initial mass does not prevent later suppression and a slower head transition.
At coefficient 0.1, the length-10 matched heads first appear positive at 2000,
with positive raw gap at 3000; both fitted by the budget. The matched length-100
head first appears positive at 5400, but its gap remains negative through
6400 and common length-100 BCE remains approximately 0.69315. Its earlier
unmatched coefficient-0.1 counterpart had positive gap at 6200 and fitted
by 6400. At coefficient 1, neither matched head nor gap becomes positive in
the observed diagnostics through 6400. These are finite-budget observations,
not evidence that later recovery is impossible.

There is a concrete local mechanism consistent with this transient. For a
positive example, write $z_+=hm+b$, where $h$ is the head slope, $m$ the target
mass, and $M$ the effective margin. Its loss derivative is

```math
\frac{\partial L_+}{\partial M}
=[\operatorname{sigmoid}(z_+)-1]h\,m(1-m).
```

When $h<0$, lowering $M$ locally reduces positive-example loss. The recorded
early updates follow that direction, with the larger coefficient producing
stronger mass suppression despite identical initial behavior. At the same
time, the positive-example gradient of the classifier's target-coordinate
weight contains a factor $m$, so that learning signal weakens as mass shrinks.
This supports a coupled attention/head transition bottleneck. It is not a
complete causal decomposition of AdamW: head coordinates, coefficient and
projections update jointly, and only the realized traces describe their
applied steps.

## Conditional factorization check

The main phase satisfied the predeclared trigger through residual fitting,
first-update and final-parameter differences. The
[decision record](../../runs/training_length/matched_initial_20261004_01/factorization_decision.json)
retains that review. The added 12 runs use seeds 3–5, coefficient 1, and each
length's own reference, comparing query-only and key-only scaling with six
reused symmetric controls. Initial scoring functions are checked against
their symmetric counterpart at all four evaluation lengths.

All 12 initial function checks passed, with maximum absolute discrepancy
1.19e-7 over corrected scores, attention, values, logits and loss across the
explicit grid. The scalar coefficient and head are unchanged across these
factorizations, while the projection norms and Jacobians differ. In exact
arithmetic their complete raw scoring functions agree at every length.

Final exponents, retaining both failed seed-5 cells:

| Seed | Train/ref | Symmetric | Query-only | Key-only |
|---:|---|---:|---:|---:|
| 3 | 10/10 | 4.3669 | 4.3455 | 4.3864 |
| 3 | 100/100 | 3.0013 | 2.9911 | 3.0245 |
| 4 | 10/10 | 4.4037 | 4.3719 | 4.4113 |
| 4 | 100/100 | 3.0076 | 2.9651 | 3.0086 |
| 5 | 10/10 | -3.8736 (failed) | -3.8984 (failed) | -3.8987 (failed) |
| 5 | 100/100 | -2.3223 (failed) | -2.3455 (failed) | -2.3457 (failed) |

The eight newly fitted runs were correct on every explicit length and crossed
strict $c\Delta>1$ at the same 200-update recording as their symmetric
counterpart, with no observed recrossing. The four failed runs had positive
accuracy zero and negative accuracy one throughout the explicit grid. This
is a null result for changing the fitting outcome with these factorizations,
not a claim of identical training dynamics or robustness to arbitrary transforms.

The first unconstrained-coefficient update is exactly equal in all 12
comparisons; its gradient differs by at most 4.66e-10. Yet the first effective
margin update relative to symmetric ranges from 0.701 to 2.036 across the
three seeds and two lengths. Both sides of the comparison retain the same
initial coefficient and scoring function, so these realized differences
expose sensitivity to the query/key factorization. Final exponent differences
from symmetric range from -0.0425 to +0.0232; fitted cells retain both the
coefficient-1 failure distinction and the length-10 exponent advantage.
The [factorization pairs](../../runs/training_length/matched_initial_20261004_01/analysis/factorization_pairs.csv)
include actual first updates and final coefficient/raw-gap differences.

![Factorization trajectories, with seed 5 dashed](../../runs/training_length/matched_initial_20261004_01/analysis/factorization_trajectories.png)

## Artifacts and validity

The [per-run table](../../runs/training_length/matched_initial_20261004_01/analysis/per_run.csv),
[initial matching](../../runs/training_length/matched_initial_20261004_01/analysis/initial_matching.csv),
[first updates](../../runs/training_length/matched_initial_20261004_01/analysis/first_updates.csv),
[window sums](../../runs/training_length/matched_initial_20261004_01/analysis/trace_windows.csv)
and [norm trajectories](../../runs/training_length/matched_initial_20261004_01/analysis/norm_trajectories.csv)
preserve all seeds and declared references. Each new run consumes 409600
balanced sequences: 4.096 million tokens at length 10 and 40.96 million at
length 100. Explicit evaluation is limited to 10, 100, 1000 and 10000;
further-length statements are analytical.

All 42 new controls retain 200-update diagnostics and 242 traced actual
updates each: 10164 trace rows, covering steps 1–200, 1590–1610 and 3190–3210.
The largest exact-update decomposition residual was 7.32e-15. Both baseline
verification runs have an additional 242 traced updates. The
[audit](../../runs/training_length/matched_initial_20261004_01/analysis/audit.json)
verifies completion, exposure, hashes, trace coverage, numerical matching and
baseline equality.

The baseline bypass plus extended observer reproduced seed 3/length 100 and
seed 5/length 10 exactly: final 6400-step weights are bitwise equal, and all
diagnostics/history are identical to the earlier runs. This includes the
1600-step prefix of 36 diagnostic rows and 50 history rows. There is no
separate existing 1600-step weight checkpoint for these seeds, so no
intermediate weight equality is claimed. The short passive-trace test also
preserves final weights, history and random states. Four new focused tests,
eight existing training-length tests, 11 Stage 3 evaluation tests, eight
Stage 4A tests and 26 Stage 4B tests passed: 57 total. The Stage 4A temporary
directory was mapped to a fixed writable workspace location for its smoke test.

Configs, initial states, trained models, source/environment manifests and
all measurements remain in the new ignored directory
`runs/training_length/matched_initial_20261004_01/`; existing run artifacts
were read as comparators. The result applies to this reduced binary detector,
the specified transforms, these seeds and this optimization budget.

Reproduce with a new output directory using `src/run_matched_initial_control.py`
and phases `main` and `factorization`; the latter receives `--symmetric-dir`
and the main phase's elapsed seconds as `--prior-training-seconds` so both share
one budget. Then run `src/analyze_matched_initial_control.py` with the common
`--run-dir` and `--factorization`. The saved manifests retain exact launch
arguments and the baseline/comparator dependencies.
