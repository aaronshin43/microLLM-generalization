# Post-fitting coefficient learning-rate branch results

All 18 paths completed at total update 6400 in 7.94
minutes on the saved CUDA environment. Six original baseline prefixes were
replayed to update 1600, then each full model/AdamW/data state was cloned into
coefficient learning-rate multipliers 1/3, 1, and 3. Seeds are 3, 4, 6; training
lengths are 10 and 100. See the [predeclared plan](COEFFICIENT_LR_BRANCH_PLAN.md)
and [execution specification](COEFFICIENT_LR_BRANCH_SPEC.md).

Follow-up execution update (2026-10-10): the held-out multiplier-2 validation
has now completed. See [multiplier-2 results](COEFFICIENT_LR_MULTIPLIER2_RESULTS.md).
The forecasts and unexecuted markers below preserve the original pre-validation
record; outcomes are recorded separately without changing those predictions.

## Main result and prediction review

The intervention changes the selected exponent while retaining successful
fitting. The predeclared endpoint prediction scorecard is below; each comparison
retains every seed rather than testing only an average. First-step numerical
checks are reported separately and do not count as long-horizon forecast success.

| Comparison | Predeclared prediction | Passed / evaluable | Censored |
|---|---|---|---|
| update6400 | fast > control > slow | 6 / 6 | 0 |
| update6400 | absolute effect larger at length10 | 6 / 6 | 0 |
| update6400 | $G_3>G_1>G_{1/3}$ | 3 / 3 | 0 |
| common100_bce0.01 | fast > control > slow | 6 / 6 | 0 |
| common100_bce0.01 | absolute effect larger at length10 | 6 / 6 | 0 |
| common100_bce0.01 | $G_3>G_1>G_{1/3}$ | 3 / 3 | 0 |

At the fixed 6400-update endpoint:

| Seed | Train length | $r$, 1/3 | $r$, 1 | $r$, 3 | $\Delta$, 1/3 | $\Delta$, 1 | $\Delta$, 3 |
|---|---|---|---|---|---|---|---|
| 3 | 10 | 0.763515 | 1.100256 | 1.889878 | 9.4549 | 9.2021 | 8.7676 |
| 3 | 100 | 0.582767 | 0.695872 | 0.969724 | 10.8647 | 10.6927 | 10.3644 |
| 4 | 10 | 0.866806 | 1.208042 | 1.953894 | 9.2704 | 9.0208 | 8.6180 |
| 4 | 100 | 0.647675 | 0.761406 | 1.025745 | 10.6470 | 10.4772 | 10.1640 |
| 6 | 10 | 0.885289 | 1.204732 | 1.916619 | 9.0571 | 8.8242 | 8.4374 |
| 6 | 100 | 0.688860 | 0.799300 | 1.052567 | 10.4329 | 10.2709 | 9.9737 |

The largest sibling difference in common length-100 balanced BCE from its
control is 1.3716831e-06, and in head slope is 0.00062704086.
Coefficient-rate changes leave a large exponent difference at very similar
loss. Faster coefficient learning is accompanied by a smaller raw score margin;
query/key compensation therefore attenuates the exponent change rather than
eliminating it. Full coefficients, logits, masses, head states, and losses are
in [final_parameters.csv](../../runs/training_length/coefficient_lr_branches_20261010_01/analysis/final_parameters.csv).

![All exponent trajectories](../../runs/training_length/coefficient_lr_branches_20261010_01/analysis/exponent_trajectories.png)

## Common evaluation loss

Parameters were interpolated at the first post-1600 downward crossing of the
same length-100 balanced BCE, between adjacent 200-update records. Primary
target: 0.01. Estimated update counts are interpolation estimates, not exact
crossing times or extra checkpoints.

| Seed | Train length | $r$, 1/3 | $r$, 1 | $r$, 3 | Update, 1/3 | Update, 1 | Update, 3 |
|---|---|---|---|---|---|---|---|
| 3 | 10 | 0.640436 | 0.787061 | 1.239314 | 3243.5 | 3241.4 | 3239.4 |
| 3 | 100 | 0.518590 | 0.568029 | 0.705881 | 3036.5 | 3036.3 | 3035.8 |
| 4 | 10 | 0.733313 | 0.889633 | 1.336734 | 3259.3 | 3257.7 | 3256.3 |
| 4 | 100 | 0.579748 | 0.631203 | 0.769935 | 3052.9 | 3052.6 | 3052.3 |
| 6 | 10 | 0.754089 | 0.900868 | 1.319548 | 3270.3 | 3268.6 | 3267.0 |
| 6 | 100 | 0.621106 | 0.673116 | 0.810869 | 3133.2 | 3133.0 | 3132.7 |

All targets, interpolation brackets, head slope/intercept/threshold, raw gap,
coefficient and attention state are retained in
[loss_matched_parameters.csv](../../runs/training_length/coefficient_lr_branches_20261010_01/analysis/loss_matched_parameters.csv).
The 0.02 target is a planned companion and 0.005 is secondary; unreached targets
are explicitly censored. Own-length loss matching is supplementary only within
each training length, since the two own-loss functions differ. Matching a scalar
loss does not equalize geometry, head state, moments or every output.

Length contrast at every planned common-loss target:

| Common BCE | Seed | $G_{1/3}$ | $G_1$ | $G_3$ | Predicted order retained |
|---|---|---|---|---|---|
| 0.02 | 3 | 0.095184 | 0.142451 | 0.302346 | True |
| 0.01 | 3 | 0.121846 | 0.219032 | 0.533433 | True |
| 0.005 | 3 | 0.147098 | 0.295780 | 0.719783 | True |
| 0.02 | 4 | 0.123553 | 0.176433 | 0.344684 | True |
| 0.01 | 4 | 0.153565 | 0.258430 | 0.566799 | True |
| 0.005 | 4 | 0.181790 | 0.337905 | 0.741148 | True |
| 0.02 | 6 | 0.104213 | 0.150754 | 0.298450 | True |
| 0.01 | 6 | 0.132984 | 0.227752 | 0.508678 | True |
| 0.005 | 6 | 0.160007 | 0.302224 | 0.676969 | True |

Paired parameter/head/step differences are in
[loss_matched_effects.csv](../../runs/training_length/coefficient_lr_branches_20261010_01/analysis/loss_matched_effects.csv).

![Exponent against common loss](../../runs/training_length/coefficient_lr_branches_20261010_01/analysis/exponent_vs_common_loss.png)

## Length contrast and what the intervention explains

Define $G_k=r_{10,k}-r_{100,k}$. Separate the gap already present at the branch
from the additional growth after branching:

| Seed | $G$ at 1600 | $G_{1/3}$ at 6400 | $G_1$ at 6400 | $G_3$ at 6400 | $G_{1/3}-G_1$ | $G_3-G_1$ |
|---|---|---|---|---|---|---|
| 3 | 0.062747 | 0.180748 | 0.404384 | 0.920154 | -0.223636 | +0.515769 |
| 4 | 0.086586 | 0.219131 | 0.446636 | 0.928149 | -0.227505 | +0.481513 |
| 6 | 0.071224 | 0.196429 | 0.405432 | 0.864052 | -0.209003 | +0.458620 |

[length_gaps_by_step.csv](../../runs/training_length/coefficient_lr_branches_20261010_01/analysis/length_gaps_by_step.csv)
reports the existing 1600 gap, later growth, and treatment-minus-control gap
at 1800, 3200 and 6400. The equal-common-loss gap and length-specific effects
are in [prediction_scorecard.csv](../../runs/training_length/coefficient_lr_branches_20261010_01/analysis/prediction_scorecard.csv).

The multiplier-1 control's later length-gap growth decomposes as follows:

| Seed | Coefficient contribution | Score contribution | Later control gap growth | Signed coefficient share |
|---|---|---|---|---|
| 3 | 0.310225 | 0.031412 | 0.341637 | 90.8% |
| 4 | 0.327094 | 0.032956 | 0.360050 | 90.8% |
| 6 | 0.300958 | 0.033251 | 0.334208 | 90.1% |

Thus coefficient updates account for approximately 90–91% of the **signed
post-1600 gap growth** under the midpoint identity. The slow and fast interventions
move the length gap in the predicted directions, and the common-loss response
persists. This combination supports the proposed post-fitting update-allocation
explanation more strongly than decomposition alone. It does not identify a
causal mediation percentage, or explain why the parent moments differed at 1600.

The substantive evidence is the direction and relative magnitude of the
predeclared length-specific response, together with its persistence at common
loss. The intervention tests solution selection **after fitting**. Parents at
the two training lengths have different optimizer/head/geometry histories;
this experiment cannot establish why the entire difference arose before 1600.

## Actual update allocation and temporal order

All six first-update forecasts were saved before any comparison tail began.
Within every parent, the step-1601 batch, initial states, gradients, Adam moments
and query/key/head parameter updates match exactly between siblings. Only the
coefficient's applied learning rate differs. Numerical first-step exponent
prediction error is zero in all 18 branches.

| Seed | Train length | Slow minus control | Fast minus control |
|---|---|---|---|
| 3 | 10 | -0.000110921 | +0.000332825 |
| 3 | 100 | -0.000048709 | +0.000146091 |
| 4 | 10 | -0.000124295 | +0.000373067 |
| 4 | 100 | -0.000052221 | +0.000156450 |
| 6 | 10 | -0.000115430 | +0.000346466 |
| 6 | 100 | -0.000050170 | +0.000150335 |

These are exponent differences after the first update, not the coefficient
updates themselves. A changed midpoint coefficient can alter the score
contribution to $r$ even while the query/key parameter updates remain identical.
The first difference thus precedes any adaptation of the query/key or head
updates. Subsequent steps allow their gradients and moments to adapt normally.

Across all 4800 actual tail updates, exact midpoint accounting gives:

```math
\delta r=\overline{\Delta}\,\delta c+\overline c\,\delta\Delta.
```

| Seed | Train length | Branch | Coefficient part | Score part | Total change in $r$ |
|---|---|---|---|---|---|
| 3 | 10 | slow | +0.145915 | +0.086963 | +0.232877 |
| 3 | 10 | control | +0.486462 | +0.083157 | +0.569619 |
| 3 | 10 | fast | +1.299129 | +0.060111 | +1.359240 |
| 3 | 100 | slow | +0.059107 | +0.055770 | +0.114877 |
| 3 | 100 | control | +0.176237 | +0.051744 | +0.227982 |
| 3 | 100 | fast | +0.461955 | +0.039879 | +0.501834 |
| 4 | 10 | slow | +0.159978 | +0.095590 | +0.255568 |
| 4 | 10 | control | +0.507842 | +0.088962 | +0.596804 |
| 4 | 10 | fast | +1.279703 | +0.062953 | +1.342656 |
| 4 | 100 | slow | +0.061924 | +0.061099 | +0.123023 |
| 4 | 100 | control | +0.180748 | +0.056006 | +0.236754 |
| 4 | 100 | fast | +0.458506 | +0.042586 | +0.501092 |
| 6 | 10 | slow | +0.152362 | +0.098612 | +0.250974 |
| 6 | 10 | control | +0.478536 | +0.091881 | +0.570417 |
| 6 | 10 | fast | +1.216093 | +0.066211 | +1.282304 |
| 6 | 100 | slow | +0.061527 | +0.064242 | +0.125769 |
| 6 | 100 | control | +0.177579 | +0.058630 | +0.236209 |
| 6 | 100 | fast | +0.444922 | +0.044553 | +0.489476 |

Signed totals are in
[all_tail_allocation.csv](../../runs/training_length/coefficient_lr_branches_20261010_01/analysis/all_tail_allocation.csv),
and differences from the control are in
[allocation_effects.csv](../../runs/training_length/coefficient_lr_branches_20261010_01/analysis/allocation_effects.csv).
The detailed 1601–1800, 3190–3210 and 6390–6400 windows retain signed **and**
absolute sums of actual updates and margin contributions in
[trace_window_allocation.csv](../../runs/training_length/coefficient_lr_branches_20261010_01/analysis/trace_window_allocation.csv).
Compact 200-update allocation files cover every tail update, not only these
detailed windows. Maximum numerical decomposition residual is
4.74e-15. Double-precision reconstruction from float32 weights
can differ slightly from the ordinary float32 diagnostic gap; this is rounding,
not a separate solution.

The accounting identity must sum correctly by construction; that alone is not
a mechanism test or causal mediation estimate. Relative length response,
gap movement, retained fitting, common-loss persistence, and frozen predictions
provide the nontrivial tests. Raw gradient size is not treated as applied update.

## Fitting, threshold entry, and extrapolation

Fitting retained at every recorded post-branch own-length point:
True. Explicit positive and negative class accuracy
is 1.0 for all paths at lengths 10, 100, 1000 and 10000:
True. Each length uses the original 50 balanced
examples. Diagnostics and model assumptions make identical-class examples
equivalent here; this is not a broad real-data accuracy estimate.

| Seed | Train length | Branch | First $r>1$ bracket | Later grid points above | Final analytical prediction |
|---|---|---|---|---|---|
| 3 | 10 | slow | Not observed | False | log10 length 17.3763 |
| 3 | 10 | control | (5200, 5400] | True | no_finite_crossing |
| 3 | 10 | fast | (2400, 2600] | True | no_finite_crossing |
| 3 | 100 | slow | Not observed | False | log10 length 11.3243 |
| 3 | 100 | control | Not observed | False | log10 length 15.2900 |
| 3 | 100 | fast | Not observed | False | log10 length 148.8805 |
| 4 | 10 | slow | Not observed | False | log10 length 30.2497 |
| 4 | 10 | control | (4000, 4200] | True | no_finite_crossing |
| 4 | 10 | fast | (2200, 2400] | True | no_finite_crossing |
| 4 | 100 | slow | Not observed | False | log10 length 13.1423 |
| 4 | 100 | control | Not observed | False | log10 length 19.0976 |
| 4 | 100 | fast | (5800, 6000] | True | no_finite_crossing |
| 6 | 10 | slow | Not observed | False | log10 length 34.3079 |
| 6 | 10 | control | (4000, 4200] | True | no_finite_crossing |
| 6 | 10 | fast | (2200, 2400] | True | no_finite_crossing |
| 6 | 100 | slow | Not observed | False | log10 length 14.5812 |
| 6 | 100 | control | Not observed | False | log10 length 22.2539 |
| 6 | 100 | fast | (5400, 5600] | True | no_finite_crossing |

Brackets refer to 200-update samples; retention means all later sampled points
remain above 1, not an assertion about unrecorded intermediate updates.
Analytical failure predictions use the final head threshold as well as $r$.
`no_finite_crossing` means the reduced two-score classifier has no positive
threshold crossing over lengths at least 2 under its exact formula. Finite
values are predicted log10 lengths, not explicit evaluations at those lengths.
No explicit evaluation beyond 10000 was run. Ranking changes are retained in
[failure_ranking_changes.csv](../../runs/training_length/coefficient_lr_branches_20261010_01/analysis/failure_ranking_changes.csv).

## Reproduction and artifacts

- Six initial hashes, 1600 diagnostic prefixes and epoch histories match the
  historical baseline exactly. No historical intermediate weight checkpoint
  exists, so intermediate bitwise weight equality is not claimed.
- Six disposable 32-update original-group versus split-group controls match
  weights, moments, batches, generator states, RNG states and history exactly.
- All six multiplier-1 final weights match the old 6400 checkpoints bitwise;
  complete diagnostic rows and epoch histories match exactly.
- Every sibling consumes the same 4800 realized tail batches. Parent checkpoint
  file hashes remain unchanged. All six sibling verification files pass.
- Six full parent bundles, 18 final models, final optimizer/data/RNG states,
  frozen forecasts, source/environment snapshots and passive traces are saved
  in the [run directory](../../runs/training_length/coefficient_lr_branches_20261010_01). Each tail has 232 detailed updates;
  each parent has 11 detailed pre-branch updates.

Scientific compute is 96000 updates: 9600 shared prefix updates plus 86400 tail
updates. Disposable grouping checks add 384 updates and first-step forecasts
add 18. Every complete comparison path represents 409600 balanced sequences;
its tail processes 307200. Actual scheduled training processes 6144000 sequences
across the unique prefixes/tails, without triple-counting the shared prefixes.
The execution stayed inside the inherited 60-minute ceiling.

The focused branching tests and existing training-length, matched-initial,
Stage 3 evaluation, Stage 4A and Stage 4B regressions pass: 61 tests. The Stage 4A
temporary directory was redirected into the workspace for sandbox compatibility.

## One selected next direction

Selected direction: `held_out_coefficient_multiplier_2`. No further experiment was run.

Test only the previously untested coefficient multiplier 2, using the same six
saved fitted parents and total budget 6400. Before that future run, the directional
prediction is $r_1<r_2<r_3$ at 6400 and common BCE 0.01; the positive difference
$r_2-r_1$ is larger at length 10 than length 100 in each selected seed, and
$G_1<G_2<G_3$. Freeze the following quantitative response model and forecasts
before running multiplier 2:

```math
g_k=r_k-r_0,\qquad
\beta=\frac{\log(g_3/g_{1/3})}{\log 9},\qquad
\widehat r_2=r_0+g_1 2^\beta.
```

Here $r_0$ is the saved parent's exponent at 1600. Fit the response separately
per parent and per comparison using only the three observed rates. This is a
phenomenological power-response hypothesis, not a derived optimization law.
Predefine numeric success as the observed $r_2-r_1$ lying within ±20% of the
predicted effect, along with retained fitting and all directional predictions.
The 20% tolerance is a declared falsification criterion, not a confidence interval
estimated from three rates. Retain a numeric miss even if the ordering succeeds.

| Seed | Train length | Predicted $r_2$, 6400 | Predicted $r_2-r_1$, 6400 | Predicted $r_2$, BCE 0.01 | Predicted $r_2-r_1$, BCE 0.01 |
|---|---|---|---|---|---|
| 3 | 10 | 1.524402 | 0.424146 | 0.992420 | 0.205359 |
| 3 | 100 | 0.830885 | 0.135013 | 0.630990 | 0.062961 |
| 4 | 10 | 1.618427 | 0.410385 | 1.099704 | 0.210071 |
| 4 | 100 | 0.893383 | 0.131977 | 0.695320 | 0.064117 |
| 6 | 10 | 1.588552 | 0.383819 | 1.096420 | 0.195551 |
| 6 | 100 | 0.925727 | 0.126427 | 0.737031 | 0.063915 |

Exact bounds and parameters are in
[held_out_multiplier2_predictions.csv](../../runs/training_length/coefficient_lr_branches_20261010_01/analysis/held_out_multiplier2_predictions.csv).
All rows are marked unexecuted. If that future experiment is authorized, freeze
these rows before launching and test them without revising the model or tolerance.

This is a held-out-rate test of the post-fitting explanation. It reuses the
initializations and does not supply independent-seed population evidence.
Relaxing the token or task assumptions remains a later decision after further
predictive validation.
