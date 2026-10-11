# Held-out coefficient multiplier 2 results

Six continuations completed through total update 6400 in 2.91
minutes. Each restores the same successful 1600-update model/AdamW/data/RNG parent
used in the [three-rate experiment](COEFFICIENT_LR_BRANCH_RESULTS.md). Only the
unconstrained coefficient parameter learning rate is 0.006; query/key and head
remain at 0.003. Seeds: 3, 4, 6. Training lengths: 10, 100.
See the [launch specification](COEFFICIENT_LR_MULTIPLIER2_SPEC.md).

Follow-up completed on 2026-10-10: the
[independent-seed validation](COEFFICIENT_LR_NEWSEEDS_RESULTS.md) tests a separately
frozen normalized-effect rule on seeds 8 and 9, without new-model rate interpolation.

## Frozen predictive test

The original forecast CSV was copied and SHA-256 checked before any full tail.
Its response-model parameters, predictions and ±20% effect tolerance were not
recomputed. Numeric success refers to the effect $r_2-r_1$, not relative error
in the absolute exponent. The original forecast file remains unchanged, including
its historical unexecuted marker; this report records the completed test separately.

- Numerical effect predictions passed: 12/12, with
  0 censored predictions.
- Predicted $r_1<r_2<r_3$ ordering passed: 12/12 comparisons.
- Larger positive effect at length 10 passed: 6/6 comparisons.
- Predicted $G_1<G_2<G_3$ ordering passed: 6/6 comparisons.
- Fitting retained on every recorded post-branch own-length point:
  True.

No model refit or tolerance adjustment was made after observing the new rate.
The largest absolute relative effect error was
13.39%.
Observed effects exceed the point forecast in 12/12
comparisons. Any systematic underprediction matters even when all stored bounds
pass; success within the tolerance does not imply an unbiased response model.

### Equal total updates: 6400

| Seed | Train length | Predicted $r_2$ | Observed $r_2$ | Predicted effect | Observed effect | Effect error | ±20% passed |
|---|---|---|---|---|---|---|---|
| 3 | 10 | 1.524402 | 1.558603 | 0.424146 | 0.458347 | +8.06% | True |
| 3 | 100 | 0.830885 | 0.846913 | 0.135013 | 0.151042 | +11.87% | True |
| 4 | 10 | 1.618427 | 1.643597 | 0.410385 | 0.435555 | +6.13% | True |
| 4 | 100 | 0.893383 | 0.908302 | 0.131977 | 0.146896 | +11.30% | True |
| 6 | 10 | 1.588552 | 1.616595 | 0.383819 | 0.411863 | +7.31% | True |
| 6 | 100 | 0.925727 | 0.940200 | 0.126427 | 0.140900 | +11.45% | True |

### Equal common length-100 balanced BCE: 0.01

| Seed | Train length | Predicted $r_2$ | Observed $r_2$ | Predicted effect | Observed effect | Effect error | ±20% passed |
|---|---|---|---|---|---|---|---|
| 3 | 10 | 0.992420 | 1.019661 | 0.205359 | 0.232600 | +13.27% | True |
| 3 | 100 | 0.630990 | 0.639422 | 0.062961 | 0.071393 | +13.39% | True |
| 4 | 10 | 1.099704 | 1.124423 | 0.210071 | 0.234790 | +11.77% | True |
| 4 | 100 | 0.695320 | 0.703790 | 0.064117 | 0.072587 | +13.21% | True |
| 6 | 10 | 1.096420 | 1.120037 | 0.195551 | 0.219168 | +12.08% | True |
| 6 | 100 | 0.737031 | 0.745560 | 0.063915 | 0.072444 | +13.34% | True |

Common-loss values are interpolated at the first post-1600 downward crossing
between adjacent 200-update records. Bounds, estimated crossing updates, raw
gap, coefficient, head state and attention diagnostics are saved in
[common_loss_parameters.csv](../../runs/training_length/coefficient_lr_multiplier2_20261010_02/analysis/common_loss_parameters.csv),
including planned companion targets 0.02 and 0.005. Scalar loss matching does
not equalize geometry, moments or the complete output state.

![Frozen numeric prediction errors](../../runs/training_length/coefficient_lr_multiplier2_20261010_02/analysis/frozen_prediction_errors.png)

## Length-specific response and update allocation

| Seed | Comparison | $A_{10}(2)$ | $A_{100}(2)$ | Larger effect at 10 | $G_1<G_2<G_3$ |
|---|---|---|---|---|---|
| 3 | update6400 | 0.458347 | 0.151042 | True | True |
| 3 | common100_bce0.01 | 0.232600 | 0.071393 | True | True |
| 4 | update6400 | 0.435555 | 0.146896 | True | True |
| 4 | common100_bce0.01 | 0.234790 | 0.072587 | True | True |
| 6 | update6400 | 0.411863 | 0.140900 | True | True |
| 6 | common100_bce0.01 | 0.219168 | 0.072444 | True | True |

The held-out rate tests the previously predicted response of the **post-fitting**
length contrast. These parents already had different geometry and optimizer
histories at 1600; this experiment does not explain the complete earlier gap.
Full control/2/3 trajectories are in
[trajectories.csv](../../runs/training_length/coefficient_lr_multiplier2_20261010_02/analysis/trajectories.csv), and length gaps in
[length_gaps.csv](../../runs/training_length/coefficient_lr_multiplier2_20261010_02/analysis/length_gaps.csv).

The exponent-effect allocation below subtracts the original multiplier-1
control's signed all-update totals from each new continuation:

| Seed | Train length | Coefficient effect | Score effect | Total exponent effect |
|---|---|---|---|---|
| 3 | 10 | +0.470215 | -0.011869 | +0.458347 |
| 3 | 100 | +0.157349 | -0.006308 | +0.151042 |
| 4 | 10 | +0.449585 | -0.014030 | +0.435555 |
| 4 | 100 | +0.154199 | -0.007304 | +0.146896 |
| 6 | 10 | +0.425605 | -0.013742 | +0.411863 |
| 6 | 100 | +0.148628 | -0.007728 | +0.140900 |

This exact midpoint accounting separates coefficient and score terms in
$\delta r=\overline\Delta\,\delta c+\overline c\,\delta\Delta$.
Raw gradients are not substituted for actual updates. The maximum numerical
decomposition residual is 4.18e-15. Double
reconstruction of float32 parameters can differ slightly from ordinary diagnostic
rounding. Detailed 232-update traces, head/projection norms, compact allocation
over all 4800 updates and first-step Adam moments are retained for each path.
The decomposition identity is not itself a causal mediation test.

## Fitting and extrapolation

| Seed | Train length | $c$ | $\Delta$ | $r_2$ | First $r>1$ bracket | Later samples above | Analytical failure |
|---|---|---|---|---|---|---|---|
| 3 | 10 | 0.174528 | 8.9304 | 1.558603 | (3000, 3200] | True | no_finite_crossing |
| 3 | 100 | 0.080668 | 10.4987 | 0.846913 | Not observed | False | log10 length 29.8253 |
| 4 | 10 | 0.187465 | 8.7675 | 1.643597 | (2600, 2800] | True | no_finite_crossing |
| 4 | 100 | 0.088261 | 10.2911 | 0.908302 | Not observed | False | log10 length 48.8096 |
| 6 | 10 | 0.188342 | 8.5833 | 1.616595 | (2600, 2800] | True | no_finite_crossing |
| 6 | 100 | 0.093140 | 10.0945 | 0.940200 | Not observed | False | log10 length 73.4064 |

Explicit positive and negative accuracy is 1.0 for all six paths at lengths
10, 100, 1000 and 10000: True. Each length
uses the original 50 balanced examples. No explicit evaluation above 10000
was performed. Finite failure values are head-aware analytical log10 length
predictions. `no_finite_crossing` is a statement under the reduced two-score
model's exact formula, not an empirical test at all lengths. Entry brackets
and retention refer to the 200-update recording grid.

## Validity and compute

All six source parent files and the original forecast file remain unchanged.
The continuation implementation hashes, Torch version and GPU match the
previous validated run. Every restored parent starts at epoch 51 / update 1600,
with original AdamW counters and moments. Six disposable first-update proposals
were saved before the full tails and reproduced the actual first updates exactly.
First states, gradients, moments, and query/key/head updates match the historical
multiplier-1 sibling exactly. Every rolling tail batch hash matches that sibling,
and the entire 50-epoch prefix history matches. No control was redefined.

Compute is 28800 valid new scientific updates, 1843200 new sequences, plus six
disposable first-update checks. Prefix training was reused without replay.
Every complete path still represents 409600 balanced sequences. The run stayed
within the inherited 60-minute execution ceiling. Five focused state-restoration,
optimizer-isolation, trace-passivity and multiplier-2 proposal tests passed;
shared training code was not changed for this experiment.

The first execution attempt failed the historical first-gradient identity gate
because deterministic CUDA kernel selection had not been enabled in the new
process. RNG restoration alone does not restore that process flag. The largest
observed key-gradient difference was 1.4551915228366852e-11. The original
`set_reproducibility` settings were then applied before parent restoration, and
all six paths were restarted into the current directory. The incomplete first
attempt is preserved and explicitly excluded from prediction scoring; it adds
9600 discarded tail updates and six disposable proposal updates
to actual compute. No forecast was changed in response to this implementation
failure. The valid attempt passes the original exact identity gates.

## Interpretation and next direction

This is a held-out learning-rate test on the same fitted parents. Numeric
prediction success strengthens the local response model only where the stored
bounds passed; preserve any miss above even if directions succeeded. The
power-response fit remains phenomenological and does not establish a universal
optimization law or independent-initialization replication.

No further training was run. Review the full prediction score before expanding
the task or model assumptions. The next scoped validation should use an independent initialization with the same task and lengths, freezing predictions before inspecting its intervention outcome. A new seed is needed to distinguish transfer of the post-fitting explanation from interpolation within these six parents. Model diversity remains a later decision.
