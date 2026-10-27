# Baseline training-length sweep results

Execution date: October 24, 2026. Nine new runs at lengths 30, 300 and 1000
completed at 6400 updates in 7.82 minutes, including
disposable checks and evaluation. Six historical baselines at lengths 10 and
100 are reused without retraining. Seeds are paired: 3, 4 and 6.
See the [plan](LENGTH_SWEEP_PLAN.md) and [launch specification](LENGTH_SWEEP_SPEC.md).

## Fitting outcomes before parameter comparisons

Final own-length fitting: **15/15** cells. All cells
remain in the tables; no seed was replaced. Correct fitting requires strictly
positive positive-class logit and strictly negative negative-class logit.
Final classifications were checked by direct forwards at every own length.

| Seed | Length | First fitting bracket | All later grid points fitted |
|---|---|---|---|
| 3 | 10 | (0, 200] | True |
| 3 | 30 | (0, 200] | True |
| 3 | 100 | (200, 400] | True |
| 3 | 300 | (200, 400] | True |
| 3 | 1000 | (200, 400] | True |
| 4 | 10 | (200, 400] | True |
| 4 | 30 | (200, 400] | True |
| 4 | 100 | (0, 200] | True |
| 4 | 300 | (200, 400] | True |
| 4 | 1000 | (200, 400] | True |
| 6 | 10 | (0, 200] | True |
| 6 | 30 | (0, 200] | True |
| 6 | 100 | (0, 200] | True |
| 6 | 300 | (0, 200] | True |
| 6 | 1000 | (200, 400] | True |

Fitting brackets and retention refer to the 200-update grid, not every update.
They do not establish permanent success beyond this budget.

## Primary: parameter selection at equal updates

| Seed | Length | Fitted | $c$ | $\Delta$ | $r$ | Own BCE | 100 BCE | Head threshold | First $r>1$ |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 10 | True | 0.119566 | 9.202075 | 1.100256 | 0.001665 | 0.001665 | 0.498273 | (5200, 5400] |
| 3 | 30 | True | 0.081857 | 9.978183 | 0.816779 | 0.001546 | 0.001546 | 0.497336 | not_observed |
| 3 | 100 | True | 0.065079 | 10.692676 | 0.695872 | 0.001457 | 0.001457 | 0.496346 | not_observed |
| 3 | 300 | True | 0.057661 | 11.266992 | 0.649671 | 0.001392 | 0.001392 | 0.495446 | not_observed |
| 3 | 1000 | True | 0.053088 | 11.837377 | 0.628422 | 0.001333 | 0.001333 | 0.494469 | not_observed |
| 4 | 10 | True | 0.133917 | 9.020840 | 1.208042 | 0.001678 | 0.001678 | 0.498303 | (4000, 4200] |
| 4 | 30 | True | 0.091609 | 9.785265 | 0.896414 | 0.001555 | 0.001555 | 0.497313 | not_observed |
| 4 | 100 | True | 0.072673 | 10.477163 | 0.761406 | 0.001466 | 0.001466 | 0.496311 | not_observed |
| 4 | 300 | True | 0.064363 | 11.006558 | 0.708414 | 0.001404 | 0.001404 | 0.495453 | not_observed |
| 4 | 1000 | True | 0.059459 | 11.517225 | 0.684803 | 0.001347 | 0.001346 | 0.494505 | not_observed |
| 6 | 10 | True | 0.136526 | 8.824194 | 1.204732 | 0.001701 | 0.001701 | 0.498857 | (4000, 4200] |
| 6 | 30 | True | 0.096156 | 9.602047 | 0.923295 | 0.001634 | 0.001635 | 0.497733 | not_observed |
| 6 | 100 | True | 0.077822 | 10.270851 | 0.799300 | 0.001560 | 0.001560 | 0.496669 | not_observed |
| 6 | 300 | True | 0.069626 | 10.782769 | 0.750759 | 0.001498 | 0.001498 | 0.495761 | not_observed |
| 6 | 1000 | True | 0.064457 | 11.285048 | 0.727401 | 0.001439 | 0.001439 | 0.494800 | not_observed |

![Parameter selection by training length](../../runs/training_length/length_sweep_20261024_01/analysis/length_response.png)

Across eligible adjacent pairs at 6400, decreasing coefficient is observed in
12/12, decreasing
exponent in 12/12,
and increasing raw margin in 12/12.
These are separate descriptive expectations, not a fitted scaling law.

| Length | $c$ min–max | $\Delta$ min–max | $r$ min–max |
|---|---|---|---|
| 10 | 0.119566–0.136526 | 8.824194–9.202075 | 1.100256–1.208042 |
| 30 | 0.081857–0.096156 | 9.602047–9.978183 | 0.816779–0.923295 |
| 100 | 0.065079–0.077822 | 10.270851–10.692676 | 0.695872–0.799300 |
| 300 | 0.057661–0.069626 | 10.782769–11.266992 | 0.649671–0.750759 |
| 1000 | 0.053088–0.064457 | 11.285048–11.837377 | 0.628422–0.727401 |

Adjacent differences are higher length minus lower length. Full paired results
at 1600 and 3200 are also retained in [adjacent_differences.csv](../../runs/training_length/length_sweep_20261024_01/analysis/adjacent_differences.csv).

| Seed | Lengths | Eligible | $\delta c$ | $\delta\Delta$ | $\delta r$ |
|---|---|---|---|---|---|
| 3 | 30 - 10 | True | -0.037710 | 0.776108 | -0.283477 |
| 3 | 100 - 30 | True | -0.016777 | 0.714493 | -0.120908 |
| 3 | 300 - 100 | True | -0.007418 | 0.574316 | -0.046201 |
| 3 | 1000 - 300 | True | -0.004573 | 0.570385 | -0.021249 |
| 4 | 30 - 10 | True | -0.042308 | 0.764425 | -0.311628 |
| 4 | 100 - 30 | True | -0.018936 | 0.691898 | -0.135008 |
| 4 | 300 - 100 | True | -0.008310 | 0.529395 | -0.052992 |
| 4 | 1000 - 300 | True | -0.004904 | 0.510667 | -0.023612 |
| 6 | 30 - 10 | True | -0.040370 | 0.777853 | -0.281437 |
| 6 | 100 - 30 | True | -0.018334 | 0.668804 | -0.123995 |
| 6 | 300 - 100 | True | -0.008196 | 0.511918 | -0.048541 |
| 6 | 1000 - 300 | True | -0.005169 | 0.502278 | -0.023358 |

![Exponent and progress trajectories](../../runs/training_length/length_sweep_20261024_01/analysis/trajectories.png)

![Coefficient, margin and head trajectories](../../runs/training_length/length_sweep_20261024_01/analysis/parameter_trajectories.png)

The [trajectory table](../../runs/training_length/length_sweep_20261024_01/analysis/trajectories.csv) includes initialization and every
200 updates, coefficient, score margin, exponent, own/common loss, head and
fitting state. [All-length diagnostics](../../runs/training_length/length_sweep_20261024_01/analysis/diagnostics_all_lengths.csv)
add effective margins, masses and logits over the declared six-length grid.
For old baselines, missing 30/300 analytical rows are calculated from their
saved state parameters; no optimizer history is inferred.

## Secondary: common length-100 balanced loss

| 100 BCE target | Reached / 15 | Fitted and eligible / 15 |
|---|---|---|
| 0.02 | 15/15 | 15/15 |
| 0.01 | 15/15 | 15/15 |
| 0.005 | 15/15 | 15/15 |

At the primary target 0.01:

| Seed | Length | Eligible | Bracket | Interpolated update | $c$ | $\Delta$ | $r$ | Own BCE | Reason |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 10 | True | (3200, 3400] | 3241.4 | 0.089376 | 8.805993 | 0.787061 | 0.009992 | eligible_fitted_crossing |
| 3 | 30 | True | (3000, 3200] | 3128.6 | 0.066169 | 9.538219 | 0.631152 | 0.009993 | eligible_fitted_crossing |
| 3 | 100 | True | (3000, 3200] | 3036.3 | 0.055496 | 10.235360 | 0.568029 | 0.010000 | eligible_fitted_crossing |
| 3 | 300 | True | (2800, 3000] | 2968.3 | 0.050747 | 10.807311 | 0.548446 | 0.010004 | eligible_fitted_crossing |
| 3 | 1000 | True | (2800, 3000] | 2907.0 | 0.047858 | 11.381795 | 0.544710 | 0.010007 | eligible_fitted_crossing |
| 4 | 10 | True | (3200, 3400] | 3257.7 | 0.102773 | 8.656143 | 0.889633 | 0.009996 | eligible_fitted_crossing |
| 4 | 30 | True | (3000, 3200] | 3141.8 | 0.075419 | 9.372255 | 0.706858 | 0.009995 | eligible_fitted_crossing |
| 4 | 100 | True | (3000, 3200] | 3052.6 | 0.062843 | 10.043981 | 0.631203 | 0.010000 | eligible_fitted_crossing |
| 4 | 300 | True | (2800, 3000] | 2986.1 | 0.057266 | 10.570108 | 0.605314 | 0.010004 | eligible_fitted_crossing |
| 4 | 1000 | True | (2800, 3000] | 2928.3 | 0.054044 | 11.086493 | 0.599160 | 0.010006 | eligible_fitted_crossing |
| 6 | 10 | True | (3200, 3400] | 3268.6 | 0.106541 | 8.455407 | 0.900868 | 0.009995 | eligible_fitted_crossing |
| 6 | 30 | True | (3200, 3400] | 3202.5 | 0.080629 | 9.202628 | 0.741998 | 0.009996 | eligible_fitted_crossing |
| 6 | 100 | True | (3000, 3200] | 3133.0 | 0.068258 | 9.861180 | 0.673116 | 0.010000 | eligible_fitted_crossing |
| 6 | 300 | True | (3000, 3200] | 3071.1 | 0.062648 | 10.374149 | 0.649922 | 0.010003 | eligible_fitted_crossing |
| 6 | 1000 | True | (3000, 3200] | 3006.9 | 0.059107 | 10.882001 | 0.643205 | 0.010005 | eligible_fitted_crossing |

Among eligible adjacent primary-loss pairs, 12/12
retain decreasing exponent. All 15 cells reach each predeclared target with
fitted own-length logits, so this primary comparison has complete coverage.
The effect does not disappear when matching this evaluation loss. At the longer
end the primary difference is small: the length-300/1000 exponent gap is much
smaller than the length-10/30 gap. The companion targets and all cells are in
[common_loss.csv](../../runs/training_length/length_sweep_20261024_01/analysis/common_loss.csv).

The earlier companion loss 0.02 has a small **reversal at 300→1000 in all three
seeds**. Coefficient still falls and raw margin rises, but their product increases
slightly there. This reverses sign again at losses 0.01 and 0.005:

| Seed | Lengths | $r$ difference at BCE 0.02 | At BCE 0.01 | At BCE 0.005 |
|---|---|---|---|---|
| 3 | 1000 - 300 | 0.003670 | -0.003736 | -0.010858 |
| 4 | 1000 - 300 | 0.001444 | -0.006155 | -0.013415 |
| 6 | 1000 - 300 | 0.001191 | -0.006717 | -0.013754 |

The primary endpoint is unchanged. This predeclared secondary observation is
not a newly chosen loss target, and it does not imply reversal at 6400 updates.

![Common-loss parameter comparison](../../runs/training_length/length_sweep_20261024_01/analysis/common_loss_response.png)

Each crossing is the first downward crossing bracketed by neighboring
200-update records from initialization, using linear interpolation of recorded
parameters and logits. This is the previous comparison convention; interpolated
parameters need not reproduce exactly the target loss under a nonlinear forward.
Eligibility requires interpolated own-length logits strictly correct.
Common loss does not match head, geometry, optimizer state or all-length behavior.

## Coefficient optimizer records and history

All nine new cells have complete 6400-row coefficient histories, fixed
200-update signed accounting blocks, and full model/AdamW/data/RNG/history
states at 1600 and 6400 after validation. Old cells have full optimizer states
at 1600/6400 and detailed windows 1590–1800, 3190–3210 and 6390–6400, plus
separately inventoried earlier gradient-only windows where available. They do
**not** have complete gradient or moment histories. See
[optimizer_availability.csv](../../runs/training_length/length_sweep_20261024_01/analysis/optimizer_availability.csv) and the immutable
reference inventory for exact step lists. No six-baseline replay was added.

Common saved-state comparison:

At common saved updates 1600, 3200 and 6400, the second moment increases with each adjacent length in 36/36 paired comparisons; the normalized direction decreases in 36/36. These are endpoint/update-state associations, not an optimizer-state intervention.

| Seed | Length | Step | First moment | Second moment | Normalized direction | Raw gradient | Actual $\delta c$ |
|---|---|---|---|---|---|---|---|
| 3 | 10 | 1600 | -9.4861e-05 | 6.13046e-07 | 0.108245 | -8.82691e-05 | 2.02358e-05 |
| 3 | 30 | 1600 | -0.000117279 | 2.00058e-06 | 0.0740822 | -0.0001091 | 1.15633e-05 |
| 3 | 100 | 1600 | -0.000141969 | 5.62978e-06 | 0.0534588 | -0.000132032 | 7.50273e-06 |
| 3 | 300 | 1600 | -0.000162929 | 1.21194e-05 | 0.0418149 | -0.000151499 | 5.56931e-06 |
| 3 | 1000 | 1600 | -0.000182961 | 2.43073e-05 | 0.0331561 | -0.000170119 | 4.27663e-06 |
| 4 | 10 | 1600 | -0.000103473 | 7.55766e-07 | 0.106342 | -0.000111102 | 2.31043e-05 |
| 4 | 30 | 1600 | -0.000125634 | 2.43149e-06 | 0.0719849 | -0.000134852 | 1.29044e-05 |
| 4 | 100 | 1600 | -0.000149523 | 6.73793e-06 | 0.0514656 | -0.00016045 | 8.22917e-06 |
| 4 | 300 | 1600 | -0.000172815 | 1.47297e-05 | 0.0402307 | -0.000185428 | 6.06477e-06 |
| 4 | 1000 | 1600 | -0.000195131 | 2.9483e-05 | 0.032108 | -0.000209372 | 4.67896e-06 |
| 6 | 10 | 1600 | -0.000120353 | 1.29915e-06 | 0.0943401 | -0.00013486 | 2.17259e-05 |
| 6 | 30 | 1600 | -0.000139992 | 3.86839e-06 | 0.0635929 | -0.000156802 | 1.23084e-05 |
| 6 | 100 | 1600 | -0.000161734 | 1.00097e-05 | 0.0456734 | -0.000181127 | 7.96095e-06 |
| 6 | 300 | 1600 | -0.000179819 | 2.01637e-05 | 0.0357786 | -0.000201413 | 5.91576e-06 |
| 6 | 1000 | 1600 | -0.000194209 | 3.71409e-05 | 0.0284719 | -0.000217489 | 4.54485e-06 |
| 3 | 10 | 6400 | -1.71682e-06 | 5.09933e-09 | 0.0240186 | -2.04629e-06 | 8.11368e-06 |
| 3 | 30 | 6400 | -2.16435e-06 | 1.65245e-08 | 0.0168216 | -2.58297e-06 | 3.97116e-06 |
| 3 | 100 | 6400 | -2.60802e-06 | 4.63654e-08 | 0.0121013 | -3.11408e-06 | 2.29478e-06 |
| 3 | 300 | 6400 | -2.97843e-06 | 9.96916e-08 | 0.00942508 | -3.55454e-06 | 1.5907e-06 |
| 3 | 1000 | 6400 | -3.33633e-06 | 1.9981e-07 | 0.00745747 | -3.98211e-06 | 1.15484e-06 |
| 4 | 10 | 6400 | -1.75387e-06 | 6.2778e-09 | 0.0221147 | -1.70825e-06 | 8.31485e-06 |
| 4 | 30 | 6400 | -2.2031e-06 | 2.00698e-08 | 0.0155372 | -2.14504e-06 | 4.09782e-06 |
| 4 | 100 | 6400 | -2.63267e-06 | 5.54699e-08 | 0.0111683 | -2.56352e-06 | 2.34693e-06 |
| 4 | 300 | 6400 | -3.03053e-06 | 1.21136e-07 | 0.00869982 | -2.9503e-06 | 1.61678e-06 |
| 4 | 1000 | 6400 | -3.43047e-06 | 2.42322e-07 | 0.00696286 | -3.34107e-06 | 1.21072e-06 |
| 6 | 10 | 6400 | -2.14006e-06 | 1.07728e-08 | 0.0205997 | -2.45833e-06 | 7.88271e-06 |
| 6 | 30 | 6400 | -2.54935e-06 | 3.19034e-08 | 0.0142602 | -2.93081e-06 | 3.91901e-06 |
| 6 | 100 | 6400 | -2.95254e-06 | 8.23727e-08 | 0.0102785 | -3.39453e-06 | 2.30223e-06 |
| 6 | 300 | 6400 | -3.27228e-06 | 1.65785e-07 | 0.00802985 | -3.76301e-06 | 1.62423e-06 |
| 6 | 1000 | 6400 | -3.57238e-06 | 3.05219e-07 | 0.00646077 | -4.10617e-06 | 1.20699e-06 |

![Optimizer measurements with availability preserved](../../runs/training_length/length_sweep_20261024_01/analysis/optimizer_records.png)

The second moment is an uncentered moving average of squared gradients, not a
centered statistical variance. A larger second moment alone does not imply a
smaller applied update: first moment, bias correction and epsilon also matter.
The actual unconstrained-parameter update and actual coefficient change are
recorded separately from the normalized direction. Complete fixed-block signed
contributions are in [allocation_blocks.csv](../../runs/training_length/length_sweep_20261024_01/analysis/allocation_blocks.csv).

Timing in the complete new histories:

| Seed | Length | Peak $\lvert g_u\rvert$ update, 1–1600 | Peak second-moment update, 1–1600 | First sampled fit | $c$ growth 0–1600 | $c$ growth 1600–6400 |
|---|---|---|---|---|---|---|
| 3 | 30 | 317 | 442 | 200 | 0.046737 | 0.028404 |
| 3 | 300 | 326 | 398 | 400 | 0.038716 | 0.012231 |
| 3 | 1000 | 336 | 398 | 400 | 0.037252 | 0.009121 |
| 4 | 30 | 362 | 488 | 400 | 0.054916 | 0.029977 |
| 4 | 300 | 381 | 442 | 400 | 0.044863 | 0.012784 |
| 4 | 1000 | 387 | 436 | 400 | 0.043130 | 0.009614 |
| 6 | 30 | 228 | 367 | 200 | 0.060011 | 0.029429 |
| 6 | 300 | 237 | 324 | 200 | 0.049960 | 0.012950 |
| 6 | 1000 | 244 | 305 | 400 | 0.047989 | 0.009753 |

Gradient and second-moment peak locations above are exact within updates 1–1600;
the fitting transition is only bracketed by the 200-update diagnostic grid.
Head slope, mass and exponent at that sampled transition are retained in
[history_timing.csv](../../runs/training_length/length_sweep_20261024_01/analysis/history_timing.csv). Coefficient growth intervals
are fixed in advance, not selected around observed peaks. These timing
associations do not establish which variable caused a change of solution.

The accounting identity is

```math
\delta r=\overline{\Delta}\,\delta c+\overline c\,\delta\Delta.
```

Maximum absolute residual across new updates:
1.1e-16. This split is an identity,
not a causal mediation percentage.

Using the actual saved squared gradients and beta, reconstruct the new cells:

```math
v_T=(1-\beta_2)\sum_{j=1}^T\beta_2^{T-j}g_{u,j}^2.
```

All 18 endpoint reconstructions pass the
predeclared tolerance. Maximum relative error is
9.38388e-05, comparing float64
history arithmetic with the recorded float32 optimizer state. Per-cell residuals
are in [moment_reconstruction.csv](../../runs/training_length/length_sweep_20261024_01/analysis/moment_reconstruction.csv).

| Seed | Length | Moment endpoint | Updates | Surviving fraction |
|---|---|---|---|---|
| 3 | 30 | 1600 | 1–200 | 0.000218744% |
| 3 | 30 | 1600 | 201–1600 | 99.9981% |
| 3 | 30 | 6400 | 1–200 | 0.000217423% |
| 3 | 30 | 6400 | 201–1600 | 99.3945% |
| 3 | 30 | 6400 | 1601–3200 | 0.430134% |
| 3 | 30 | 6400 | 3201–6400 | 0.16735% |
| 3 | 300 | 1600 | 1–200 | 9.89507e-07% |
| 3 | 300 | 1600 | 201–1600 | 99.9986% |
| 3 | 300 | 6400 | 1–200 | 9.87611e-07% |
| 3 | 300 | 6400 | 201–1600 | 99.8069% |
| 3 | 300 | 6400 | 1601–3200 | 0.13394% |
| 3 | 300 | 6400 | 3201–6400 | 0.051306% |
| 3 | 1000 | 1600 | 1–200 | 6.45126e-08% |
| 3 | 1000 | 1600 | 201–1600 | 99.9983% |
| 3 | 1000 | 6400 | 1–200 | 6.4433e-08% |
| 3 | 1000 | 6400 | 201–1600 | 99.8749% |
| 3 | 1000 | 6400 | 1601–3200 | 0.0837699% |
| 3 | 1000 | 6400 | 3201–6400 | 0.0319212% |
| 4 | 30 | 1600 | 1–200 | 8.93507e-05% |
| 4 | 30 | 1600 | 201–1600 | 99.9985% |
| 4 | 30 | 6400 | 1–200 | 8.88733e-05% |
| 4 | 30 | 6400 | 201–1600 | 99.4642% |
| 4 | 30 | 6400 | 1601–3200 | 0.385128% |
| 4 | 30 | 6400 | 3201–6400 | 0.142881% |
| 4 | 300 | 1600 | 1–200 | 4.13923e-07% |
| 4 | 300 | 1600 | 201–1600 | 99.9986% |
| 4 | 300 | 6400 | 1–200 | 4.13221e-07% |
| 4 | 300 | 6400 | 201–1600 | 99.8292% |
| 4 | 300 | 6400 | 1601–3200 | 0.118308% |
| 4 | 300 | 6400 | 3201–6400 | 0.0438694% |
| 4 | 1000 | 1600 | 1–200 | 2.6888e-08% |
| 4 | 1000 | 1600 | 201–1600 | 99.9986% |
| 4 | 1000 | 6400 | 1–200 | 2.68584e-08% |
| 4 | 1000 | 6400 | 201–1600 | 99.8887% |
| 4 | 1000 | 6400 | 1601–3200 | 0.0749633% |
| 4 | 1000 | 6400 | 3201–6400 | 0.0277164% |
| 6 | 30 | 1600 | 1–200 | 6.55257% |
| 6 | 30 | 1600 | 201–1600 | 93.4457% |
| 6 | 30 | 6400 | 1–200 | 6.52301% |
| 6 | 30 | 6400 | 201–1600 | 93.0242% |
| 6 | 30 | 6400 | 1601–3200 | 0.320875% |
| 6 | 30 | 6400 | 3201–6400 | 0.124372% |
| 6 | 300 | 1600 | 1–200 | 0.261199% |
| 6 | 300 | 1600 | 201–1600 | 99.737% |
| 6 | 300 | 6400 | 1–200 | 0.260818% |
| 6 | 300 | 6400 | 201–1600 | 99.5917% |
| 6 | 300 | 6400 | 1601–3200 | 0.1002% |
| 6 | 300 | 6400 | 3201–6400 | 0.0388231% |
| 6 | 1000 | 1600 | 1–200 | 0.0268608% |
| 6 | 1000 | 1600 | 201–1600 | 99.9717% |
| 6 | 1000 | 6400 | 1–200 | 0.0268352% |
| 6 | 1000 | 6400 | 201–1600 | 99.8762% |
| 6 | 1000 | 6400 | 1601–3200 | 0.0634561% |
| 6 | 1000 | 6400 | 3201–6400 | 0.0245857% |

Fractions above are surviving contributions to the endpoint second moment.
They do not measure causal importance or the fraction of learning accomplished.
The old-length history gap is retained, rather than filled by sparse interpolation.

Across the nine new cells, updates 201–1600 supply
93.45%–99.9986%
of the second moment at 1600. Updates 1–200 supply at most
6.55% (seed 6, length 30),
so the previous tiny-first-200 observation for old seeds 3/4 must not be
generalized unchanged to every new cell. At update 6400, the surviving combined
contribution of updates 1–1600 is still 99.39%–99.90%
of the remaining second moment. Early raw-gradient peaks occur at updates
228–387,
and the early second-moment peaks at
305–488.
Thus the new histories localize much of the recorded moment accumulation to
the early fitting transition. They do not establish its causal effect on the
final exponent or reconstruct the old lengths' missing history.

## Finite evaluation and analytical extrapolation

| Seed | Length | Head-aware status | Analytical failure log10 length | All direct grid classes correct | Sampled exits from $r>1$ |
|---|---|---|---|---|---|
| 3 | 10 | no_finite_crossing | unavailable | True | 0 |
| 3 | 30 | finite | 23.677 | True | 0 |
| 3 | 100 | finite | 15.290 | True | 0 |
| 3 | 300 | finite | 13.990 | True | 0 |
| 3 | 1000 | finite | 13.861 | True | 0 |
| 4 | 10 | no_finite_crossing | unavailable | True | 0 |
| 4 | 30 | finite | 41.071 | True | 0 |
| 4 | 100 | finite | 19.098 | True | 0 |
| 4 | 300 | finite | 16.421 | True | 0 |
| 4 | 1000 | finite | 15.899 | True | 0 |
| 6 | 10 | no_finite_crossing | unavailable | True | 0 |
| 6 | 30 | finite | 54.417 | True | 0 |
| 6 | 100 | finite | 22.254 | True | 0 |
| 6 | 300 | finite | 18.818 | True | 0 |
| 6 | 1000 | finite | 18.012 | True | 0 |

Both classes were directly evaluated at 10, 30, 100, 300, 1000 and 10000.
Historical direct evaluations are retained on shared lengths; old final weights
were evaluated only at missing lengths 30/300, without training. Their measured
masses and logits agree with the corresponding analytical diagnostics within
the declared numerical tolerance. [Explicit evaluations](../../runs/training_length/length_sweep_20261024_01/analysis/explicit_evaluations.csv)
preserve provenance. All claims beyond 10000 and finite failure log10 lengths
are head-aware predictions of the reduced two-score model, not direct tests.
Neither a larger exponent nor a sampled threshold entry establishes convergence
or a universal length law.

## Compute and validity

New scientific training: 57600 updates,
3686400 sequence exposures and 1634304000
token exposures. Disposable passivity checks: 384
additional updates across six 64-update paths. New-run validation adds 921600
sequence exposures; final new-model direct evaluation adds 2700 examples and
saved-weight evaluations add 600 examples. Historical training/validation is
reused, not newly charged. [Exposure records](../../runs/training_length/length_sweep_20261024_01/analysis/realized_exposure.csv)
retain per-length counts. Equal updates equalize sequence exposures, while
token exposure and runtime grow with length.

The passivity checks separately process 24576 training sequence exposures
(10895360 tokens) and 6144 validation sequence exposures (2723840 tokens).
Scientific validation processes 408576000 tokens; new-model final evaluation
processes 1716000 tokens and saved-weight evaluation 99000 tokens. These are
additional validation/evaluation costs, excluded from the scientific training budget.

The six references matched their verified control weights, complete diagnostics
and epoch histories. Same-seed initial state hashes match at all five lengths.
Observed/unobserved short paths at every new length match weights, optimizer,
RNG, loaders and history exactly. Full saved states are captured after validation;
all historical source hashes and frozen source snapshots remain unchanged.
The manifest records resolved dtype, CUDA/deterministic settings and optimizer
options. All 15 cells and all nine complete new coefficient traces are retained.

The full automated suite recorded 94 tests, with passed status
True, including history reconstruction and
loss-crossing/censoring checks. Its temporary files were routed to the ignored
workspace validation directory after the system temporary directory denied writes.
Tiny CPU training paths inside unit tests are separate from scientific exposure
accounting. The suite log is retained in `validation/full_test_suite.txt`.

## Direction review

The fixed-update and primary common-loss exponent ordering both extend to all five lengths, with a flattening response at the long end. However, the predeclared BCE-0.02 companion reveals a small length-300/1000 reversal in all three seeds, while the sign is negative at BCE 0.01 and 0.005. The selected next action is a focused analysis of this progress-dependent 300/1000 contrast, using the saved coefficient/score/head and optimizer records to formulate a testable account before selecting a second-moment intervention. The small reversal is based on the 200-update interpolation convention and needs that resolution caveat. Do not treat a universal monotone loss-matched law as established. No further training or optimizer intervention was launched.
