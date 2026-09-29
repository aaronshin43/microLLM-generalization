# Learned-Log 6400-Update Confirmation Results

All nine fresh runs completed in 3.76 minutes.
The [launch specification](CONFIRMATION_6400_SPEC.md) fixes the original three
training conditions, seeds and model, changing only the budget to 6400 updates.
All runs restarted from seed initialization with a new optimizer. At 1600 updates,
**all nine** matched pilot weights bit for bit, all 36 diagnostic rows exactly,
and all history rows through that budget exactly. No checkpoint weights were
loaded to continue optimization. See [prefix verification](../../runs/training_length/confirmation_6400_20260928_01/analysis/prefix_verification.json).
The six related tests passed, including unchanged final weights, RNG and history
with the intermediate verification/checkpoint enabled, independent failure-formula
checks, head-validity checks and loss-interpolation checks.

## Answers to the four questions

1. **The mixture's exponent advantage over `[100]` persists.** It is positive
   in every seed at every recorded point from 1600 through 6400 updates. The paired
   mean difference grows from about 0.015 to 0.026. The mixture still has a smaller
   raw margin, so this is not a simple uniform improvement in attention parameters.
2. **Only `[10]` enters the above-1 regime within this budget.** Seeds 0 and 1
   first appear above 1 at 5600 updates, seed 2 at 4400, and remain above at every
   later recorded point through 6400. Neither `[100]` nor the mixture crosses by 6400.
3. **Finite failure-length rankings reverse before threshold entry.** At 1600,
   all seeds rank `[100] > [10,100] > [10]`; by 6400 all rank `[10] > [10,100] > [100]`.
   Mixture overtakes `[100]` at the first recorded points 2200, 6200 and 2600 for
   seeds 0, 1 and 2 respectively. These are sampled ranking-change times, not exact
   continuous crossing times. The change demonstrates why final exponents alone
   do not describe the entire training trajectory.
4. **A scalar progress-speed explanation is insufficient for the observed gaps.**
   `[100]` reaches shared length-100 loss targets sooner than the mixture, yet has
   a smaller exponent. At matched common length-100 BCE, mixture-minus-100 exponent
   differences remain about 0.028–0.031 while raw-margin differences remain negative.
   This is descriptive evidence of different parameter allocation at comparable
   measured performance, not proof of convergence or a uniquely identified cause.

## Parameter trajectories and the small exponent advantage

Means ± sample standard deviations across the three paired seeds:

| Updates | Training lengths | $c$ | $\Delta$ | $c\Delta$ | Balanced own-objective BCE |
|---|---|---:|---:|---:|---:|
| 1600 | [10] | 0.0729 ± 0.0060 | 8.014 ± 0.297 | 0.583 ± 0.027 | 0.03534 ± 0.00253 |
| 1600 | [100] | 0.0572 ± 0.0026 | 9.463 ± 0.218 | 0.541 ± 0.013 | 0.03169 ± 0.00199 |
| 1600 | [10,100] | 0.0629 ± 0.0029 | 8.843 ± 0.237 | 0.556 ± 0.011 | 0.03610 ± 0.00261 |
| 6400 | [10] | 0.1243 ± 0.0125 | 8.960 ± 0.356 | 1.111 ± 0.068 | 0.00162 ± 0.00010 |
| 6400 | [100] | 0.0744 ± 0.0030 | 10.338 ± 0.237 | 0.768 ± 0.014 | 0.00145 ± 0.00007 |
| 6400 | [10,100] | 0.0811 ± 0.0037 | 9.794 ± 0.264 | 0.794 ± 0.015 | 0.00162 ± 0.00009 |

Mixture minus `[100]` differences (paired across seeds):

| Updates | $\Delta$ difference | $c\Delta$ difference | Per-seed exponent differences (0,1,2) |
|---|---:|---:|---|
| 1600 | -0.620 ± 0.023 | 0.015 ± 0.011 | +0.0221, +0.0021, +0.0218 |
| 3200 | -0.583 ± 0.030 | 0.021 ± 0.012 | +0.0278, +0.0072, +0.0275 |
| 4800 | -0.561 ± 0.032 | 0.024 ± 0.012 | +0.0303, +0.0099, +0.0305 |
| 6400 | -0.544 ± 0.034 | 0.026 ± 0.012 | +0.0321, +0.0120, +0.0328 |

![Parameter and loss trajectories](../../runs/training_length/confirmation_6400_20260928_01/analysis/confirmation_trajectories.png)

The gray vertical line marks 1600 updates. The plots show changing parameters
and both the native training objective and the common length-100 diagnostic.
Analytical balanced BCE uses the exact two-score model; measured batch/validation
losses remain in each run's `train_history.csv`.

## First growth-threshold entry and persistence

Strictly above $c\Delta=1$ is recorded at 200-update intervals. The intervals
bracket a crossing between observations; they do not locate an exact update.
Persistence means all later observations **through 6400**, not indefinitely.

| Seed | Training lengths | First crossing interval | All later observations above 1 | Final $c\Delta$ |
|---|---|---|---|---:|
| 0 | [10] | (5400, 5600] | True | 1.0726 |
| 0 | [100] | Not observed by 6400 | False | 0.7524 |
| 0 | [10,100] | Not observed by 6400 | False | 0.7846 |
| 1 | [10] | (5400, 5600] | True | 1.0710 |
| 1 | [100] | Not observed by 6400 | False | 0.7739 |
| 1 | [10,100] | Not observed by 6400 | False | 0.7859 |
| 2 | [10] | (4200, 4400] | True | 1.1894 |
| 2 | [100] | Not observed by 6400 | False | 0.7788 |
| 2 | [10,100] | Not observed by 6400 | False | 0.8116 |

## Predicted failure-length rankings

Ranks run from longest predicted failure length to shortest. `>` describes the
head-aware analytical prediction, not measured long-sequence accuracy. A verified
no-finite-threshold-crossing case ranks above finite predictions; tied no-crossing
cases are grouped. Rankings begin only once all three conditions classify their
training classes correctly and their heads support this comparison.

| Seed | First observation of this ranking | Longest predicted failure first |
|---|---:|---|
| 0 | 400 | [100] > [10,100] > [10] |
| 0 | 2000 | [10] > [100] > [10,100] |
| 0 | 2200 | [10] > [10,100] > [100] |
| 1 | 400 | [100] > [10,100] > [10] |
| 1 | 2000 | [100] > [10] > [10,100] |
| 1 | 2200 | [10] > [100] > [10,100] |
| 1 | 6200 | [10] > [10,100] > [100] |
| 2 | 400 | [100] > [10,100] > [10] |
| 2 | 1800 | [10] > [100] > [10,100] |
| 2 | 2600 | [10] > [10,100] > [100] |

![Analytical failure-length trajectories](../../runs/training_length/confirmation_6400_20260928_01/analysis/failure_length_trajectories.png)

Values beyond $\log_{10} n=60$ are clipped only in the plot; the full values
are in [failure rankings](../../runs/training_length/confirmation_6400_20260928_01/analysis/failure_rankings_by_step.csv).
No-finite-crossing cases are omitted from the finite curves. The solver checks
head slope/threshold and, when the exponent exceeds 1, the minimum attention gap;
an exponent alone is not treated as a universal classification criterion.

## Does matching progress remove the parameter difference?

Match the first downward crossing of a shared loss using linear interpolation
between adjacent 200-update recordings. This is approximate descriptive matching,
not extra training or causal identification. `objective_bce` uses each condition's
own balanced length distribution; `common100_bce` evaluates every model at length
100, so its definition is shared across conditions. Compare both rather than
equating unlike training losses.

| Matched loss | Target BCE | Estimated updates: mixture minus 100 | $\Delta$ difference | $c\Delta$ difference |
|---|---:|---:|---:|---:|
| objective_bce | 0.04 | 120.4 ± 11.3 | -0.555 ± 0.027 | 0.028 ± 0.012 |
| objective_bce | 0.02 | 149.7 ± 12.8 | -0.554 ± 0.030 | 0.029 ± 0.012 |
| objective_bce | 0.01 | 176.5 ± 13.8 | -0.548 ± 0.032 | 0.030 ± 0.012 |
| objective_bce | 0.005 | 198.6 ± 15.5 | -0.540 ± 0.034 | 0.031 ± 0.012 |
| common100_bce | 0.04 | 121.6 ± 10.9 | -0.555 ± 0.027 | 0.028 ± 0.012 |
| common100_bce | 0.02 | 150.7 ± 12.6 | -0.554 ± 0.030 | 0.029 ± 0.012 |
| common100_bce | 0.01 | 177.3 ± 13.7 | -0.548 ± 0.032 | 0.030 ± 0.012 |
| common100_bce | 0.005 | 199.2 ± 15.4 | -0.540 ± 0.034 | 0.031 ± 0.012 |

For the common loss definition, the exponent levels across all three conditions are:

| Matched length-100 BCE | `[10]` $c\Delta$ | `[100]` $c\Delta$ | `[10,100]` $c\Delta$ |
|---|---:|---:|---:|
| 0.04 | 0.559 ± 0.010 | 0.517 ± 0.010 | 0.545 ± 0.003 |
| 0.02 | 0.692 ± 0.023 | 0.584 ± 0.010 | 0.613 ± 0.006 |
| 0.01 | 0.818 ± 0.034 | 0.641 ± 0.011 | 0.671 ± 0.008 |
| 0.005 | 0.935 ± 0.045 | 0.691 ± 0.011 | 0.722 ± 0.010 |

Full interpolation brackets, per-seed parameters and differences are in
[loss-matched parameters](../../runs/training_length/confirmation_6400_20260928_01/analysis/loss_matched_parameters.csv) and
[paired differences](../../runs/training_length/confirmation_6400_20260928_01/analysis/loss_matched_paired_differences.csv).
Persistence of a parameter gap at matched loss argues against describing it only
as a scalar training-speed shift; it does not prove a unique mechanism or rule out
all optimization/transient explanations. Loss curves still changing at 6400 limit
claims about stable limiting parameter selection.

## Exposure, explicit evaluation and reproducibility

Every run processed 409600 sequences, balanced by class. Mixtures used 204800
sequences per length. Token totals were 4096000 (`[10]`), 40960000 (`[100]`) and
22528000 (`[10,100]`). Same maximum length and updates do not match token budget
or length-100 exposure. Recorded counts are [here](../../runs/training_length/confirmation_6400_20260928_01/analysis/realized_exposure.csv).

Final explicit sequence evaluation remained 10, 100, 1000, 10000, 50 examples per
length. All 36 run/length attention checks matched the independent two-score
formula within `rtol=3e-5`, `atol=3e-6`. Beyond that grid, including failure lengths,
values are closed-form predictions; no 10M sequence evaluation was performed.

Each run saves final and step-1600 model-only checkpoints, history, 132 diagnostic
rows (33 times × 4 lengths), exposure and verification JSON. No optimizer state is
saved. [Manifest](../../runs/training_length/confirmation_6400_20260928_01/manifest.json), `source_snapshot/`,
[run summary](../../runs/training_length/confirmation_6400_20260928_01/pilot_summary.csv), and [trajectory summary](../../runs/training_length/confirmation_6400_20260928_01/analysis/trajectory_summary.csv)
preserve settings, code/environment and all seeds. Outputs remain ignored/local;
the original pilot and report artifacts were read only.
