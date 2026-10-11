# Independent-seed coefficient-rate results

Eight paths for seeds 8 and 9 completed at total update 6400 in
4.40 minutes. Four shared prefixes were trained through
1600, then full model/AdamW/data/RNG states were cloned into coefficient-rate
multipliers 1 and 2. Training lengths remained 10 and 100. Query/key and head
learning rates stayed 0.003. See the [launch specification](COEFFICIENT_LR_NEWSEEDS_SPEC.md).
No new-seed 1/3 or 3 multiplier was run, and no seed was replaced.

## Parent eligibility

| Seed | Train length | Fitted at 1600 | Positive logit | Negative logit | $r$ at branch |
|---|---|---|---|---|---|
| 8 | 10 | True | +3.5338 | -3.4707 | 0.510366 |
| 8 | 100 | True | +3.7083 | -3.5475 | 0.452383 |
| 9 | 10 | True | +3.3137 | -3.2541 | 0.555731 |
| 9 | 100 | True | +3.5101 | -3.3822 | 0.501824 |

Eligibility includes both own-length classes strictly correct, positive head
slope and positive raw margin. Every intended parent is retained in the table.
Failed eligibility would make its pair descriptive rather than a confirmation
of a post-fitting prediction; it would count against full four-parent success.

## Frozen rule and numerical predictions

The existing seed-3/4/6 effect divided by its control's post-branch exponent
growth was averaged separately by length and comparison. This normalized-effect
rule is distinct from the previous power-response fit. Its four coefficients
were hashed before any new-seed training: 0.7521685749951815 / 0.6264930352686643
at 6400, and 0.8575632252231385 / 0.6842067538150549 at common BCE 0.01, for
training lengths 10 / 100 respectively. No coefficient was refitted per new seed.

All four new control paths completed before any multiplier-2 tail. Their
post-branch growth was the only new outcome used to construct the stored effect
forecast. All eight forecasts and ±20% effect bounds were saved before any
multiplier-2 training. The exact source hashes and calibration data remain unchanged.
This is a conditional prediction given the new control path, not a prediction
of that control's optimization trajectory from initialization.

Primary endpoint, update 6400: 4/4
evaluable forecasts passed, out of four planned pairs.

| Seed | Train length | Predicted $r_2$ | Observed $r_2$ | Predicted effect | Observed effect | Effect error | Passed |
|---|---|---|---|---|---|---|---|
| 8 | 10 | 1.549497 | 1.577345 | 0.446077 | 0.473925 | +6.24% | True |
| 8 | 100 | 0.816807 | 0.826709 | 0.140369 | 0.150271 | +7.05% | True |
| 9 | 10 | 1.519184 | 1.539681 | 0.413590 | 0.434087 | +4.96% | True |
| 9 | 100 | 0.873595 | 0.875592 | 0.143199 | 0.145196 | +1.39% | True |

Secondary endpoint, common length-100 balanced BCE 0.01:
4/4 evaluable forecasts passed,
out of four planned pairs.

| Seed | Train length | Predicted $r_2$ | Observed $r_2$ | Predicted effect | Observed effect | Effect error | Passed |
|---|---|---|---|---|---|---|---|
| 8 | 10 | 0.925854 | 0.940530 | 0.191814 | 0.206491 | +7.65% | True |
| 8 | 100 | 0.592222 | 0.595433 | 0.056809 | 0.060020 | +5.65% | True |
| 9 | 10 | 1.021363 | 1.026422 | 0.214964 | 0.220023 | +2.35% | True |
| 9 | 100 | 0.669419 | 0.670093 | 0.068085 | 0.068759 | +0.99% | True |

Common-loss parameters use the first post-branch downward crossing interpolated
between adjacent 200-update records. The same predeclared ±20% criterion applies
to the effect $r_2-r_1$, not the absolute exponent. Censored or inapplicable
forecasts remain in [numeric_predictions.csv](../../runs/training_length/coefficient_lr_newseeds_20261010_01/analysis/numeric_predictions.csv)
with reasons. No tolerance, target or seed was adjusted after the outcome.

![Independent-seed prediction errors](../../runs/training_length/coefficient_lr_newseeds_20261010_01/analysis/independent_prediction_errors.png)

## Direction, fitting, and extrapolation

| Seed | Comparison | Effect at 10 | Effect at 100 | Larger at 10 |
|---|---|---|---|---|
| 8 | update6400 | 0.473925 | 0.150271 | True |
| 8 | common100_bce0.01 | 0.206491 | 0.060020 | True |
| 9 | update6400 | 0.434087 | 0.145196 | True |
| 9 | common100_bce0.01 | 0.220023 | 0.068759 | True |

Positive exponent effect passed 8/8 planned
comparisons. Larger effect at length 10 passed 4/4.
Own-length fitting retained at every recorded post-branch point:
True. All explicit positive/negative accuracies at
10, 100, 1000, 10000 are 1.0: True.

| Seed | Train length | Rate | $r$ | First $r>1$ bracket | Later grid above | Analytical failure |
|---|---|---|---|---|---|---|
| 8 | 10 | 1 | 1.103420 | (5200, 5400] | True | no_finite_crossing |
| 8 | 10 | 2 | 1.577345 | (3000, 3200] | True | no_finite_crossing |
| 8 | 100 | 1 | 0.676438 | Not observed | False | log10 length 13.893 |
| 8 | 100 | 2 | 0.826709 | Not observed | False | log10 length 25.466 |
| 9 | 10 | 1 | 1.105594 | (5000, 5200] | True | no_finite_crossing |
| 9 | 10 | 2 | 1.539681 | (3000, 3200] | True | no_finite_crossing |
| 9 | 100 | 1 | 0.730396 | Not observed | False | log10 length 16.849 |
| 9 | 100 | 2 | 0.875592 | Not observed | False | log10 length 35.867 |

Threshold-entry brackets and retention refer to the 200-update grid. Larger
lengths and failure log10 lengths are analytical predictions under the reduced
two-score classifier, including head threshold; no explicit evaluation beyond
10000 was performed. Full parameters, head state, common/own losses, exposure,
232 detailed updates per tail and compact signed allocation across all 4800
tail updates are retained. This confirms only the specified seeds and model.

## Why length 10 responds more strongly: local optimizer analysis

Analyze recorded states without another intervention. At update 1601, sibling
weights, batch, gradients and Adam moments match; query/key/head parameter
updates match. Only the coefficient learning rate differs. For the first
coefficient-rate intervention, a first-order factorization is

```math
A_L(2,1601)\approx\Delta_L^+\,\sigma(u_L)\,\eta_u D_L,
\qquad D_L=-\frac{\widehat m_L}{\sqrt{\widehat v_L}+\epsilon}.
```

Here $\sigma(u)$ is the softplus derivative, $\eta_u=0.003$ is the baseline
coefficient learning rate, and $D$ is the actual Adam normalized direction.
The finite first-step difference is also recorded directly, using float32
optimizer updates rather than assuming a perfect multiplier-to-step equality.
Maximum relative error of the linearized response across old and new parents:
0.108%.

| Seed | $\Delta^+_{10}/\Delta^+_{100}$ | Softplus derivative ratio | Adam direction ratio | Raw gradient magnitude ratio | Variance 100 / 10 | First response 10 / 100 |
|---|---|---|---|---|---|---|
| 3 | 0.8446 | 1.3318 | 2.0250 | 0.6686 | 9.1832 | 2.2779 |
| 4 | 0.8479 | 1.3602 | 2.0664 | 0.6925 | 8.9153 | 2.3856 |
| 6 | 0.8438 | 1.3219 | 2.0657 | 0.7446 | 7.7048 | 2.3060 |
| 8 | 0.8208 | 1.3623 | 2.1417 | 0.7757 | 7.6321 | 2.3947 |
| 9 | 0.8503 | 1.2922 | 2.0239 | 0.6677 | 9.2000 | 2.2234 |

Across all five seeds, length 10 has a **smaller** raw margin and a **smaller**
instantaneous coefficient-gradient magnitude. Those quantities alone therefore
do not explain its larger first-step response. Its softplus derivative is
1.29–1.36 times larger, and its applied Adam normalized direction is 2.02–2.14
times larger. The length-100 coefficient second moment is 7.63–9.20 times larger;
its larger normalization denominator outweighs its larger current gradient and
first moment. Together the three factors reproduce the 2.22–2.39 response ratio.

As a conditional arithmetic check, retain the measured first moments, raw
margins and softplus derivatives but make the two Adam denominators equal:

| Seed | First-moment magnitude 10 / 100 | Local response ratio if denominators equal |
|---|---|---|
| 3 | 0.6682 | 0.7517 |
| 4 | 0.6921 | 0.7982 |
| 6 | 0.7442 | 0.8301 |
| 8 | 0.7753 | 0.8669 |
| 9 | 0.6673 | 0.7332 |

Each such local ratio is below 1. This is **not** a trained moment-matching
intervention or a causal estimate. It shows that the observed denominator
asymmetry is necessary to the greater length-10 response in this particular
first-step factorization; it does not prove that holding moments equal would
preserve fitting or the later optimization path.

The table compares length 10 with length 100 for each seed. It separates three
factors that are otherwise hidden in a raw-gradient comparison: raw margin,
softplus sensitivity, and Adam normalization. Geometry enters through the raw
margin; query/key norms and their cosine are also retained in
[local_optimizer_states.csv](../../runs/training_length/coefficient_lr_newseeds_20261010_01/analysis/local_optimizer_states.csv).
The factorization is a local conditional explanation at a fitted checkpoint,
not an identification of how all parent differences arose during the first 1600
updates. Subsequent query/key/head adaptation can attenuate or amplify the effect;
all-update control-minus-treatment allocation is available for that distinction.

For this two-token task, the negative class contributes no analytical coefficient
gradient. With positive batch fraction $\rho$, positive attention mass $p$,
positive logit $z$ and head slope $h$, the coefficient gradient follows

```math
g_u=\rho[\sigma(z)-1]h\,p(1-p)\Delta\log(1+L)\sigma(u).
```

Its maximum relative discrepancy from the recorded first-step gradient is
0.0135%, consistent with numerical
rounding. This distinguishes attention/head/length effects on the raw gradient
from the optimizer's applied update. The current gradient and moment are not
interchangeable; the second moment remembers earlier squared gradients.

Existing old-seed traces cover every update 1–200 for seeds 3 and 4. Under the
unchanged Adam settings, their surviving contribution to variance at 1600 is

```math
v_{1600,1:200}=(1-\beta_2)\sum_{j=1}^{200}
\beta_2^{1600-j}g_j^2,\qquad\beta_2=0.999.
```

| Seed | Train length | Fraction of step-1600 variance from updates 1–200 |
|---|---|---|
| 3 | 10 | 0.00309856% |
| 3 | 100 | 1.26122e-05% |
| 4 | 10 | 0.00114985% |
| 4 | 100 | 5.34079e-06% |

The rest of the 1600-update variance is not reconstructed from sparse traces.
The first 200 updates account for at most 0.0031% of the step-1600 second moment
in these four old-parent cells. Thus the very earliest 1–200 gradients do **not**
explain the large variance difference. The relevant squared-gradient history
was accumulated predominantly during updates 201–1600, which are not fully
covered by the existing detailed traces. This rules out an earliest-phase
memory explanation and identifies the unresolved fitting-transition interval.
This early-history analysis covers only old seeds 3 and 4; it is not imputed
for seed 6 or new seeds 8/9. Source hashes and contributions are retained in
[early_gradient_memory.csv](../../runs/training_length/coefficient_lr_newseeds_20261010_01/analysis/early_gradient_memory.csv).

The complete post-branch exponent effect also shows compensating score updates:

| Seed | Train length | Coefficient contribution difference | Score contribution difference | Total exponent effect |
|---|---|---|---|---|
| 8 | 10 | +0.486223 | -0.012298 | +0.473925 |
| 8 | 100 | +0.156303 | -0.006032 | +0.150271 |
| 9 | 10 | +0.445936 | -0.011849 | +0.434087 |
| 9 | 100 | +0.151963 | -0.006767 | +0.145196 |

The negative score contribution attenuates the positive coefficient contribution
in all new pairs. This exact signed decomposition is an accounting identity,
not a causal mediation percentage. Head changes are recorded alongside it;
the first head update is identical across siblings, while later head adaptation
remains allowed. Endpoint head/geometry states are not assumed to match merely
because the common loss matches.

For the successful new pairs, factor the endpoint effect ratio into control
growth and normalized response, using the same denominator as the frozen rule:

| Seed | Comparison | Control-growth ratio | Normalized-response ratio | Actual effect ratio |
|---|---|---|---|---|
| 8 | update6400 | 2.6469 | 1.1915 | 3.1538 |
| 8 | common100_bce0.01 | 2.6939 | 1.2771 | 3.4404 |
| 9 | update6400 | 2.4056 | 1.2428 | 2.9897 |
| 9 | common100_bce0.01 | 2.5190 | 1.2703 | 3.1999 |

All ratios are length 10 divided by length 100. Much of the larger endpoint
effect comes from a larger baseline post-branch exponent increase; a further
length-specific normalized-response difference contributes. This is a
descriptive factorization, not an explanation of how the baseline growth or
moment history originated.

## Validity, compute, and next direction

All four full parent files remain unchanged. Same-seed initialization hashes
match across lengths. Four disposable 32-update original/split-group comparisons
match weights, moments, history, RNG, generators and batches exactly. All four
new sibling pairs have identical initial gradients/moments, first query/key/head
updates, full future batch hashes and prefix histories. Frozen rule and numerical
forecast hashes pass. Five serialized-continuation/isolation tests passed again.
Three additional tests verify the frozen growth-based forecast, failed-parent
handling and censoring without extrapolation.
No shared training code was modified. The process uses the original deterministic
kernel settings explicitly.

Scientific compute is 44800 updates and 2867200 sequence exposures, counting
four shared prefixes once and each tail. Sibling tails deliberately reuse batches.
Disposable grouping checks add 256 updates. Each
complete path represents 409600 balanced sequence exposures. Execution remains inside
the inherited 60-minute ceiling.

Overall four-parent confirmation, including eligibility, both numeric comparisons,
positive effects, larger length-10 response and retained fitting:
True. This independent-initialization result tests
transfer of a length-specific rule without new-model rate interpolation. It
does not establish a universal optimization law from two new seeds.

The next direction is explanation of the length response using geometry and
optimizer history, rather than expanding learning-rate choices. The read-only
analysis above starts that work by isolating immediate softplus/Adam amplification
and checking early-gradient memory on the available old traces. Future causal
controls would need to separate these factors without silently changing initial
behavior or optimizer history. No additional training or model-assumption change
was run after the eight paths.
