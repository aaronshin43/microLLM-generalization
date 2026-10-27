# Training-length continuation to 12800 updates

Execution date: October 24, 2026. All nine baseline continuations reached total
update 12800 from their original full 6400-update states in
6.37 minutes, including audit, probes and evaluation.
Lengths: 10, 30, 1000. Paired seeds: 3, 4, 6. See the
[plan](LENGTH_EXTENSION_12800_PLAN.md) and [launch specification](LENGTH_EXTENSION_12800_SPEC.md).

## Fitting and threshold attainment

Final own-length fitting: **9/9**. Fitting retained at every
200-update extension diagnostic: True.
Both classes are correct at every final direct evaluation length through 10000:
True. No path or seed was replaced.

New threshold entries: length 30, **3/3**;
length 1000, **0/3**. Length 10 was already above
1 at extension start. Successful fitting and threshold attainment are distinct.

| Seed | Length | Start status | Historical entry grid bracket | First new entry | Censored at 12800 | Retained above | Exits |
|---|---|---|---|---|---|---|---|
| 3 | 10 | already_above_at_start | (5200, 5400] | not_observed | False | True | 0 |
| 4 | 10 | already_above_at_start | (4000, 4200] | not_observed | False | True | 0 |
| 6 | 10 | already_above_at_start | (4000, 4200] | not_observed | False | True | 0 |
| 3 | 30 | entered_during_extension | not_observed | 10939 | False | True | 0 |
| 4 | 30 | entered_during_extension | not_observed | 8809 | False | True | 0 |
| 6 | 30 | entered_during_extension | not_observed | 8188 | False | True | 0 |
| 3 | 1000 | right_censored_12800 | not_observed | not_observed | True | False | 0 |
| 4 | 1000 | right_censored_12800 | not_observed | not_observed | True | False | 0 |
| 6 | 1000 | right_censored_12800 | not_observed | not_observed | True | False | 0 |

New entry/exit steps use each actual update, with detached float64 query/key
dot-product margin and the actual float32 softplus coefficient. Historical
length-10 entries remain grid-bracketed; 6401 is not a new entry merely because
a trace starts there. Censoring means no new entry within this fixed budget.
The diagnostic series retains its original float32 score-subtraction definition.
Maximum absolute trace-versus-grid exponent difference:
2.12562e-07; these numerical conventions
are not silently mixed to select entry times.

## Fixed-update solutions and paired gaps

| Seed | Length | Step | $c$ | $\Delta$ | $r$ | Own BCE | 100 BCE | Head slope | Head threshold | Fitted |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 10 | 6400 | 0.119566 | 9.202075 | 1.100256 | 0.001665 | 0.001665 | 12.795112 | 0.498273 | True |
| 3 | 10 | 9600 | 0.142494 | 9.406771 | 1.340405 | 0.000335 | 0.000335 | 16.005970 | 0.498731 | True |
| 3 | 10 | 12800 | 0.161649 | 9.541332 | 1.542348 | 0.000070 | 0.000070 | 19.139938 | 0.499038 | True |
| 4 | 10 | 6400 | 0.133917 | 9.020840 | 1.208042 | 0.001678 | 0.001678 | 12.779754 | 0.498303 | True |
| 4 | 10 | 9600 | 0.157050 | 9.212599 | 1.446843 | 0.000337 | 0.000337 | 15.991619 | 0.498749 | True |
| 4 | 10 | 12800 | 0.176149 | 9.340338 | 1.645289 | 0.000070 | 0.000070 | 19.125821 | 0.499065 | True |
| 6 | 10 | 6400 | 0.136526 | 8.824194 | 1.204732 | 0.001701 | 0.001701 | 12.752782 | 0.498857 | True |
| 6 | 10 | 9600 | 0.158946 | 9.023446 | 1.434240 | 0.000342 | 0.000342 | 15.963563 | 0.499176 | True |
| 6 | 10 | 12800 | 0.177510 | 9.157818 | 1.625605 | 0.000071 | 0.000071 | 19.098207 | 0.499371 | True |
| 3 | 30 | 6400 | 0.081857 | 9.978183 | 0.816779 | 0.001546 | 0.001546 | 12.944684 | 0.497336 | True |
| 3 | 30 | 9600 | 0.093062 | 10.213722 | 0.950508 | 0.000311 | 0.000311 | 16.153583 | 0.498054 | True |
| 3 | 30 | 12800 | 0.102536 | 10.379066 | 1.064232 | 0.000065 | 0.000065 | 19.286241 | 0.498520 | True |
| 4 | 30 | 6400 | 0.091609 | 9.785265 | 0.896414 | 0.001555 | 0.001555 | 12.933598 | 0.497313 | True |
| 4 | 30 | 9600 | 0.102985 | 10.008944 | 1.030774 | 0.000312 | 0.000312 | 16.143514 | 0.498033 | True |
| 4 | 30 | 12800 | 0.112502 | 10.167152 | 1.143824 | 0.000065 | 0.000065 | 19.276326 | 0.498516 | True |
| 6 | 30 | 6400 | 0.096156 | 9.602047 | 0.923295 | 0.001634 | 0.001635 | 12.833101 | 0.497733 | True |
| 6 | 30 | 9600 | 0.107296 | 9.828163 | 1.054525 | 0.000329 | 0.000329 | 16.042021 | 0.498363 | True |
| 6 | 30 | 12800 | 0.116585 | 9.989491 | 1.164628 | 0.000069 | 0.000069 | 19.175309 | 0.498752 | True |
| 3 | 1000 | 6400 | 0.053088 | 11.837377 | 0.628422 | 0.001333 | 0.001333 | 13.245463 | 0.494469 | True |
| 3 | 1000 | 9600 | 0.056284 | 12.080494 | 0.679934 | 0.000269 | 0.000269 | 16.449701 | 0.495947 | True |
| 3 | 1000 | 12800 | 0.058878 | 12.261557 | 0.721936 | 0.000056 | 0.000056 | 19.578708 | 0.496895 | True |
| 4 | 1000 | 6400 | 0.059459 | 11.517225 | 0.684803 | 0.001347 | 0.001346 | 13.225211 | 0.494505 | True |
| 4 | 1000 | 9600 | 0.062755 | 11.748931 | 0.737303 | 0.000271 | 0.000271 | 16.430652 | 0.495971 | True |
| 4 | 1000 | 12800 | 0.065436 | 11.922676 | 0.780171 | 0.000057 | 0.000057 | 19.559930 | 0.496922 | True |
| 6 | 1000 | 6400 | 0.064457 | 11.285048 | 0.727401 | 0.001439 | 0.001439 | 13.091284 | 0.494800 | True |
| 6 | 1000 | 9600 | 0.067807 | 11.510733 | 0.780508 | 0.000290 | 0.000290 | 16.295243 | 0.496213 | True |
| 6 | 1000 | 12800 | 0.070513 | 11.680444 | 0.823620 | 0.000061 | 0.000061 | 19.424727 | 0.497101 | True |

![Combined coefficient, margin and exponent trajectories](../../runs/training_length/length_extension_12800_20261024_01/analysis/combined_trajectories.png)

![Length response at 6400 and 12800](../../runs/training_length/length_extension_12800_20261024_01/analysis/length_response.png)

The tentative endpoint ordering passed 3/3 seeds.
Over the extension diagnostic grid it held in
99/99
paired comparisons; intermediate ordering was measured rather than assumed.
Threshold crossing does not mean the coefficient, margin or exponent gaps vanish.

Both paired endpoint gaps **widened in 3/3 seeds**:
the increase in $r_{10}-r_{30}$ is
0.179539–0.194640,
and in $r_{30}-r_{1000}$ it is
0.145114–0.153939.
Thus longer training achieves a threshold for length 30 while preserving and
expanding the measured equal-update separation of the successful solutions.

| Seed | Step | $r_{10}-r_{30}$ | Change since 6400 | $r_{30}-r_{1000}$ | Change since 6400 | Ordering |
|---|---|---|---|---|---|---|
| 3 | 6400 | 0.283477 | 0.000000 | 0.188357 | 0.000000 | True |
| 3 | 9600 | 0.389897 | 0.106421 | 0.270574 | 0.082217 | True |
| 3 | 12800 | 0.478116 | 0.194640 | 0.342296 | 0.153939 | True |
| 4 | 6400 | 0.311628 | 0.000000 | 0.211611 | 0.000000 | True |
| 4 | 9600 | 0.416069 | 0.104441 | 0.293471 | 0.081860 | True |
| 4 | 12800 | 0.501465 | 0.189837 | 0.363653 | 0.152042 | True |
| 6 | 6400 | 0.281437 | 0.000000 | 0.195893 | 0.000000 | True |
| 6 | 9600 | 0.379715 | 0.098278 | 0.274018 | 0.078124 | True |
| 6 | 12800 | 0.460977 | 0.179539 | 0.341008 | 0.145114 | True |

Average exponent growth per update in the three predeclared 1600-update windows:

| Seed | Length | 4800–6400 growth/update | 8000–9600 | 11200–12800 |
|---|---|---|---|---|
| 3 | 10 | 8.969146e-05 | 7.1305102e-05 | 6.069618e-05 |
| 3 | 30 | 5.018717e-05 | 3.974633e-05 | 3.4328555e-05 |
| 3 | 1000 | 2.0062706e-05 | 1.5077033e-05 | 1.2640572e-05 |
| 4 | 10 | 9.0728999e-05 | 7.0592824e-05 | 5.9521865e-05 |
| 4 | 30 | 5.1074459e-05 | 3.9793474e-05 | 3.4051944e-05 |
| 4 | 1000 | 2.0616131e-05 | 1.5320205e-05 | 1.2881946e-05 |
| 6 | 10 | 8.7012291e-05 | 6.7904002e-05 | 5.744785e-05 |
| 6 | 30 | 5.0040939e-05 | 3.8829644e-05 | 3.3142929e-05 |
| 6 | 1000 | 2.0995527e-05 | 1.5354293e-05 | 1.2947286e-05 |

![Paired gaps and fixed-window growth](../../runs/training_length/length_extension_12800_20261024_01/analysis/gaps_growth.png)

These finite-window changes describe progress, not convergence or a guaranteed
linear threshold forecast. In all nine paths the final window's average growth
is smaller than the pre-extension window: its ratio is
0.617–0.684. There is continued growth alongside slowing,
so neither a permanent failure claim nor an unchanged-rate crossing forecast
follows for length 1000. [Full trajectories](../../runs/training_length/length_extension_12800_20261024_01/analysis/trajectories.csv) retain
initialization, all 200-update diagnostics, head state, masses, logits and
own/common loss. The 6400 boundary is included once. The original prefix is
reused; only missing length-10 analytical 30/300 rows were added from saved
parameters, without inferring earlier optimizer history.

![Own-length and common evaluation loss](../../runs/training_length/length_extension_12800_20261024_01/analysis/loss_trajectories.png)

## New common-loss comparisons

| Length-100 BCE | Reached / 9 | Fitted and eligible / 9 |
|---|---|---|
| 0.001 | 9/9 | 9/9 |
| 0.0005 | 9/9 | 9/9 |
| 0.0001 | 9/9 | 9/9 |

All three declared targets are reached by all nine fitted paths. The exponent
ordering also remains strict in every paired common-loss triplet:

| 100 BCE target | Seed | $r_{10}-r_{30}$ | $r_{30}-r_{1000}$ | Ordering |
|---|---|---|---|---|
| 0.001 | 3 | 0.325382 | 0.216746 | True |
| 0.001 | 4 | 0.353877 | 0.240288 | True |
| 0.001 | 6 | 0.319138 | 0.225309 | True |
| 0.0005 | 3 | 0.370721 | 0.251713 | True |
| 0.0005 | 4 | 0.398121 | 0.275016 | True |
| 0.0005 | 6 | 0.360929 | 0.258438 | True |
| 0.0001 | 3 | 0.464329 | 0.327226 | True |
| 0.0001 | 4 | 0.488817 | 0.349203 | True |
| 0.0001 | 6 | 0.447201 | 0.328992 | True |

Thus the length contrast is present at matched evaluation loss as well as at
equal updates. This does not prove independence from every aspect of optimization
progress, since only one evaluation loss is matched.

Target 0.001:

| Seed | Length | Eligible | Crossing bracket | Interpolated update | $c$ | $\Delta$ | $r$ | Head threshold | Reason |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 10 | True | (7200, 7400] | 7399.3 | 0.127254 | 9.277363 | 1.180586 | 0.498436 | fitted_crossing |
| 4 | 10 | True | (7400, 7600] | 7414.5 | 0.141835 | 9.092077 | 1.289579 | 0.498408 | fitted_crossing |
| 6 | 10 | True | (7400, 7600] | 7442.8 | 0.144394 | 8.899710 | 1.285071 | 0.499016 | fitted_crossing |
| 3 | 30 | True | (7200, 7400] | 7255.3 | 0.085084 | 10.051289 | 0.855204 | 0.497603 | fitted_crossing |
| 4 | 30 | True | (7200, 7400] | 7265.8 | 0.094943 | 9.855388 | 0.935703 | 0.497489 | fitted_crossing |
| 6 | 30 | True | (7200, 7400] | 7364.8 | 0.099784 | 9.680210 | 0.965933 | 0.497989 | fitted_crossing |
| 3 | 1000 | True | (6800, 7000] | 6962.8 | 0.053712 | 11.886721 | 0.638458 | 0.494811 | fitted_crossing |
| 4 | 1000 | True | (6800, 7000] | 6981.6 | 0.060126 | 11.565877 | 0.695415 | 0.494831 | fitted_crossing |
| 6 | 1000 | True | (7000, 7200] | 7115.2 | 0.065293 | 11.343069 | 0.740624 | 0.495173 | fitted_crossing |

Target 0.0005:

| Seed | Length | Eligible | Crossing bracket | Interpolated update | $c$ | $\Delta$ | $r$ | Head threshold | Reason |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 10 | True | (8600, 8800] | 8787.1 | 0.137096 | 9.363586 | 1.283712 | 0.498630 | fitted_crossing |
| 4 | 10 | True | (8800, 9000] | 8801.0 | 0.151733 | 9.172653 | 1.391792 | 0.498674 | fitted_crossing |
| 6 | 10 | True | (8800, 9000] | 8830.3 | 0.153981 | 8.983294 | 1.383254 | 0.499111 | fitted_crossing |
| 3 | 30 | True | (8600, 8800] | 8639.9 | 0.089925 | 10.152729 | 0.912990 | 0.497882 | fitted_crossing |
| 4 | 30 | True | (8600, 8800] | 8650.3 | 0.099851 | 9.951512 | 0.993671 | 0.497886 | fitted_crossing |
| 6 | 30 | True | (8600, 8800] | 8752.4 | 0.104570 | 9.776469 | 1.022325 | 0.498220 | fitted_crossing |
| 3 | 1000 | True | (8200, 8400] | 8344.1 | 0.055128 | 11.995305 | 0.661278 | 0.495448 | fitted_crossing |
| 4 | 1000 | True | (8200, 8400] | 8362.9 | 0.061585 | 11.669268 | 0.718655 | 0.495503 | fitted_crossing |
| 6 | 1000 | True | (8400, 8600] | 8499.4 | 0.066761 | 11.442156 | 0.763888 | 0.495809 | fitted_crossing |

Target 0.0001:

| Seed | Length | Eligible | Crossing bracket | Interpolated update | $c$ | $\Delta$ | $r$ | Head threshold | Reason |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 10 | True | (12000, 12200] | 12065.9 | 0.157516 | 9.514446 | 1.498681 | 0.498980 | fitted_crossing |
| 4 | 10 | True | (12000, 12200] | 12080.5 | 0.172123 | 9.315211 | 1.603360 | 0.498994 | fitted_crossing |
| 6 | 10 | True | (12000, 12200] | 12108.8 | 0.173747 | 9.132354 | 1.586718 | 0.499365 | fitted_crossing |
| 3 | 30 | True | (11800, 12000] | 11916.9 | 0.100052 | 10.338162 | 1.034351 | 0.498418 | fitted_crossing |
| 4 | 30 | True | (11800, 12000] | 11926.9 | 0.110042 | 10.128360 | 1.114543 | 0.498414 | fitted_crossing |
| 6 | 30 | True | (12000, 12200] | 12029.1 | 0.114471 | 9.954606 | 1.139517 | 0.498715 | fitted_crossing |
| 3 | 1000 | True | (11600, 11800] | 11617.7 | 0.057964 | 12.199313 | 0.707125 | 0.496591 | fitted_crossing |
| 4 | 1000 | True | (11600, 11800] | 11637.6 | 0.064509 | 11.863978 | 0.765339 | 0.496676 | fitted_crossing |
| 6 | 1000 | True | (11600, 11800] | 11775.3 | 0.069692 | 11.630095 | 0.810525 | 0.496861 | fitted_crossing |

![Post-6400 common-loss exponent comparison](../../runs/training_length/length_extension_12800_20261024_01/analysis/common_loss.png)

Every target is below the source losses. Crossings are the first post-6400
downward crossing between adjacent 200-update records. Linear interpolation
uses the established parameter/logit convention; a nonlinear forward on the
interpolated parameters need not reproduce the target exactly. Eligibility
requires strictly correct own-length logits. Censored cells remain visible;
there is no extrapolation or substitution of an earlier sweep target.
Common loss does not match the head, geometry, optimizer state or complete function.

## Optimizer memory and actual updates

The complete new trace has 6400 updates per path, 57600 in total, with raw
coefficient gradients, actual unconstrained-parameter/coefficient/exponent
changes, before/after moments, bias correction and normalized applied direction.
See [moment observations](../../runs/training_length/length_extension_12800_20261024_01/analysis/moment_observations.csv) and each run's complete
`coefficient_trace.csv` and `allocation_by_200.csv`. Maximum signed-accounting
residual is 2.21528e-16 for

```math
\delta r=\overline{\Delta}\,\delta c+\overline c\,\delta\Delta.
```

This is an accounting identity, not causal mediation. Adam's second moment is
an uncentered moving average of squared gradients, not centered variance.
Boundary and new-gradient contributions use

```math
v_T=\beta_2^{T-6400}v_{6400}
 +(1-\beta_2)\sum_{j=6401}^T\beta_2^{T-j}g_{u,j}^2.
```

| Seed | Length | Step | Recorded second moment | Pre-6400 fraction | New-gradient fraction | Relative residual | Passed |
|---|---|---|---|---|---|---|---|
| 3 | 10 | 9600 | 2.07792e-10 | 99.8724% | 0.1235% | 4.05128e-05 | True |
| 3 | 10 | 12800 | 8.46236e-12 | 99.8036% | 0.1885% | 7.92111e-05 | True |
| 4 | 10 | 9600 | 2.55762e-10 | 99.8927% | 0.1034% | 3.91959e-05 | True |
| 4 | 10 | 12800 | 1.04149e-11 | 99.8341% | 0.1580% | 7.95932e-05 | True |
| 6 | 10 | 9600 | 4.3885e-10 | 99.9019% | 0.0943% | 3.78637e-05 | True |
| 6 | 10 | 12800 | 1.78696e-11 | 99.8474% | 0.1449% | 7.70857e-05 | True |
| 3 | 30 | 9600 | 6.72951e-10 | 99.9328% | 0.0631% | 4.00602e-05 | True |
| 3 | 30 | 12800 | 2.73981e-11 | 99.8927% | 0.0995% | 7.84968e-05 | True |
| 4 | 30 | 9600 | 8.17246e-10 | 99.9430% | 0.0531% | 3.85268e-05 | True |
| 4 | 30 | 12800 | 3.32713e-11 | 99.9076% | 0.0836% | 8.77274e-05 | True |
| 6 | 30 | 9600 | 1.29903e-09 | 99.9491% | 0.0468% | 4.14214e-05 | True |
| 6 | 30 | 12800 | 5.28841e-11 | 99.9162% | 0.0738% | 9.99572e-05 | True |
| 3 | 1000 | 9600 | 8.13278e-09 | 99.9861% | 0.0127% | 1.18947e-05 | True |
| 3 | 1000 | 12800 | 3.30996e-10 | 99.9813% | 0.0202% | 1.54152e-05 | True |
| 4 | 1000 | 9600 | 9.86279e-09 | 99.9897% | 0.0109% | 5.96857e-06 | True |
| 4 | 1000 | 12800 | 4.01404e-10 | 99.9853% | 0.0174% | 2.7388e-05 | True |
| 6 | 1000 | 9600 | 1.24224e-08 | 99.9925% | 0.0097% | 2.16079e-05 | True |
| 6 | 1000 | 12800 | 5.05575e-10 | 99.9884% | 0.0154% | 3.84278e-05 | True |

Reconstruction passes: **18/18**,
using the unchanged relative tolerance 0.0001 and absolute tolerance 1e-12.
Maximum relative residual: 9.99572e-05.
Fractions are against the recorded float32 state; their sum can differ slightly
from 1 due to the retained float64 reconstruction residual.

![Surviving pre-6400 moment contribution](../../runs/training_length/length_extension_12800_20261024_01/analysis/moment_memory.png)

At 12800 the saved-boundary contribution supplies
99.8036%–99.9884%
of the remaining recorded second moment. Its absolute contribution has itself
decayed to 0.1656% of its saved boundary magnitude. A high surviving
fraction therefore means the new squared-gradient contribution is small relative
to that decayed memory; it does not mean the second moment stayed constant.
Coefficient/exponent growth continues while average exponent growth slows in
the fixed windows, and applied updates still depend on first moment, bias
correction and the softplus parameterization. This combination motivates a
controlled test, rather than establishing causal importance from fractions.

For lengths 30/1000, split that inherited history further without changing its
exact saved-boundary contribution. Values below are surviving moment quantities:

| Seed | Length | Step | Inherited 1–1600 | Inherited 1601–6400 | Boundary reconstruction residual |
|---|---|---|---|---|---|
| 3 | 10 | 9600 | missing earlier history | missing earlier history | missing earlier history |
| 3 | 10 | 12800 | missing earlier history | missing earlier history | missing earlier history |
| 4 | 10 | 9600 | missing earlier history | missing earlier history | missing earlier history |
| 4 | 10 | 12800 | missing earlier history | missing earlier history | missing earlier history |
| 6 | 10 | 9600 | missing earlier history | missing earlier history | missing earlier history |
| 6 | 10 | 12800 | missing earlier history | missing earlier history | missing earlier history |
| 3 | 30 | 9600 | 6.68428e-10 | 4.01807e-12 | 5.23979e-14 |
| 3 | 30 | 12800 | 2.7203e-11 | 1.63523e-13 | 2.13244e-15 |
| 4 | 30 | 9600 | 8.12405e-10 | 4.31268e-12 | 6.3309e-14 |
| 4 | 30 | 12800 | 3.30624e-11 | 1.75513e-13 | 2.57648e-15 |
| 6 | 30 | 9600 | 1.29249e-09 | 5.78096e-12 | 9.8545e-14 |
| 6 | 30 | 12800 | 5.26006e-11 | 2.35268e-13 | 4.01049e-15 |
| 3 | 1000 | 9600 | 8.12148e-09 | 9.40759e-12 | 7.63064e-13 |
| 3 | 1000 | 12800 | 3.3052e-10 | 3.82861e-13 | 3.10544e-14 |
| 4 | 1000 | 9600 | 9.8508e-09 | 1.0126e-11 | 8.54414e-13 |
| 4 | 1000 | 12800 | 4.00898e-10 | 4.121e-13 | 3.47721e-14 |
| 6 | 1000 | 9600 | 1.24094e-08 | 1.09361e-11 | 1.10861e-12 |
| 6 | 1000 | 12800 | 5.05027e-10 | 4.45066e-13 | 4.5117e-14 |

Length 10's complete earlier gradient history is missing; its boundary carryover
is identifiable and is not split using sparse interpolation. The extra split
for lengths 30/1000 preserves its inherited reconstruction error separately.
A high carryover fraction is not a causal attribution, and a changed fraction
alone does not explain the selected exponent or head adaptation.

## Direct evaluation and analytical extrapolation

| Seed | Length | $r$ at 12800 | Head-aware analytical status | Predicted failure log10 length |
|---|---|---|---|---|
| 3 | 10 | 1.542348 | no_finite_crossing | unavailable |
| 4 | 10 | 1.645289 | no_finite_crossing | unavailable |
| 6 | 10 | 1.625605 | no_finite_crossing | unavailable |
| 3 | 30 | 1.064232 | no_finite_crossing | unavailable |
| 4 | 30 | 1.143824 | no_finite_crossing | unavailable |
| 6 | 30 | 1.164628 | no_finite_crossing | unavailable |
| 3 | 1000 | 0.721936 | finite | 19.170 |
| 4 | 1000 | 0.780171 | finite | 23.579 |
| 6 | 1000 | 0.823620 | finite | 28.789 |

The six final direct evaluation lengths are 10, 30, 100, 300, 1000 and 10000.
Both classes, logits and positive attention masses are retained in
[explicit_evaluations.csv](../../runs/training_length/length_extension_12800_20261024_01/analysis/explicit_evaluations.csv) and agree with the
reduced-model formulas within the declared numerical tolerance. All failures
beyond 10000, including no-finite-crossing status, are head-aware analytical
predictions for fixed trained parameters, not direct 10M tests.

## Restoration, compute and tests

All source files and normalized starting bundles remain hash-identical.
Length-10 final weights/history/diagnostics matched the verified historical
baseline before restoration. Integer optimizer IDs were checked against named
moments; two source groups at length 10 and one at lengths 30/1000 were preserved,
including all options and counters. Recreated dataset hashes, loader generators,
Python/NumPy/Torch RNG and weights matched every source. The first new epoch
is 201; source validation was not repeated. Saved states at 9600/12800 include
model/optimizer/data/RNG/history after completed validation.

Seed-3 serialization/passivity probes at every length compare 64 uninterrupted
unobserved updates with 32 observed updates, save/load, and 32 further observed
updates. All final weights, optimizer, batches, RNG/loaders and histories match
exactly. No uninterrupted 0–12800 historical run exists; these are restoration
and serialization checks, not a claim of comparison to one.

Scientific continuation compute: 57600 new updates,
3686400 new sequence exposures and
1277952000 new token exposures. Each cumulative path has
819200 sequence exposures. Prefix training is reused, not newly charged.
Probes add 384 updates, 24576 training sequence exposures
and 8519680 tokens; probe validation adds 6144 exposures and 2129920 tokens.
Scientific validation adds 921600 exposures and 319488000 tokens. Final direct
evaluation adds 2700 examples and 1716000 tokens. Unit-test CPU paths are separate.
Equal updates equalize sequence exposures, while token exposure/runtime differ.

Full automated suite: 97 tests, passed status
True. The log is `validation/full_test_suite.txt`.
Workspace-local temporary files avoid the known system temporary-directory
write restriction. [Audit](../../runs/training_length/length_extension_12800_20261024_01/analysis/audit.json), source inventory, manifest, state
hashes, exact group mappings and snapshots retain full provenance.

## Direction review

Length 30 attained the threshold in all three seeds with the larger budget, while length 1000 did not. The paired exponent gaps widened and persisted at the reached common losses. The selected next action is a focused comparison of the saved length-30/1000 growth, head and moment records before designing a limited optimizer-state intervention. Non-entry is right-censored at 12800 for the listed paths; it does not imply permanent failure or convergence. Do not extend the budget until all paths cross.

This extension does not establish a universal training-length law, permanent
threshold retention, convergence, or the optimizer history's causal role.
