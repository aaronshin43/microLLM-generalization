# Stage 3D Multiple Non-Target Tokens

## Objective

Stage 3D tests whether the reduced Stage 3 length-aware attention model still works when there are several non-target token types.

The original Stage 3 setup relied on the two-score assumption: one target score and one shared non-target score. Stage 3D intentionally weakens that assumption by adding multiple non-target token types:

```math
t,\ u_1,\ u_2,\ \ldots,\ u_m.
```

The main question is:

**Can length-aware attention still generalize when non-target scores are not all identical?**

## Generalized Theory

The Stage 3D model still computes ordinary position-level softmax attention. The generalized formula below is not a new attention mechanism. It is an algebraic rewrite of the same softmax that groups repeated non-target positions by token type, so that we can analyze which non-target types dominate the denominator.

### Original Two-Score Case

The original Stage 3 setup had one target token and one non-target token. A positive input produced the final-query attention score row:

```math
S_n=(a,b,b,\ldots,b).
```

Here:

- $S_n$ is the final-query attention score row for a length-$n$ positive input.
- $a$ is the target key score.
- $b$ is the shared non-target key score.
- $\Delta=a-b$ is the target-vs-non-target score margin.

This makes the target attention mass reduce to:

```math
p_t(n)
=
\frac{e^{\alpha a}}
{e^{\alpha a}+(n-1)e^{\alpha b}}
=
\frac{1}
{1+(n-1)e^{-\alpha(a-b)}}.
```

### Multiple Non-Target Case

In Stage 3D, the last token is no longer always the same non-target token. Because the model reads from the final query, the attention score row depends on which non-target token appears at the final position.

For a positive sequence:

```math
t, u_{i_1}, u_{i_2}, ..., u_{i_{n-1}}
```

let $r=i_{n-1}$ be the final non-target token type. Then the target score and non-target scores are conditional on $r$:

```math
S_n^{(r)}=(a_r,b_{r,i_1},b_{r,i_2},\ldots,b_{r,i_{n-1}}).
```

Here:

- $r\in\{1,\ldots,m\}$ is the last-token type used to form the final query.
- $a_r$ is the target key score under final query type $r$.
- $b_{r,k}$ is the score for non-target token type $u_k$ under final query type $r$.
- $c_k(n)$ is the count of non-target token type $u_k$ in a length-$n$ positive input.
- $\alpha$ is the attention score multiplier at the current sequence length.

Then:

```math
\sum_{k=1}^{m} c_k(n)=n-1.
```

The target attention mass is:

```math
p_t(n\mid r)
=
\frac{e^{\alpha a_r}}
{e^{\alpha a_r}+\sum_{k=1}^{m}c_k(n)e^{\alpha b_{r,k}}}.
```

Divide by $e^{\alpha a_r}$:

```math
p_t(n\mid r)
=
\frac{1}
{1+\sum_{k=1}^{m}c_k(n)e^{-\alpha(a_r-b_{r,k})}}.
```

Define the per-type margin:

```math
\Delta_{r,k}=a_r-b_{r,k}.
```

Then:

```math
p_t(n\mid r)
=
\frac{1}
{1+\sum_{k=1}^{m}c_k(n)e^{-\alpha\Delta_{r,k}}}.
```

The bottleneck is the smallest margin:

```math
\Delta_{\min}=\min_{r,k}\Delta_{r,k}.
```

A smaller $\Delta_{r,k}$ means that non-target type $u_k$ is scored closer to the target under final query type $r$, so its denominator term decays more slowly and can dominate at long length.

For learned-log attention,

```math
\alpha=1+c\log(1+n),
```

the asymptotic condition is approximately:

```math
c\min_{r,k}\Delta_{r,k}>1.
```

This is the Stage 3D analogue of the original Stage 3 condition $c\Delta>1$.

## Setup

This report analyzes the Stage 3D main condition:

```text
non_target_token_count = 4
train_lengths = [10]
test_examples = 50
eval_batch_size = 8
eval_lengths = 10, 100, 1000, 10000, 100000, 1000000, 5000000, 10000000
```

Token ids:

- target token: `0`
- non-target tokens: `1`, `2`, `3`, `4`

Positive examples contain exactly one target token at position 0:

```math
t, u_{i_1}, u_{i_2}, ..., u_{i_{n-1}}
```

Negative examples contain only non-target tokens:

```math
u_{i_1}, u_{i_2}, ..., u_{i_n}
```

Non-target tokens are sampled uniformly from token ids `1` through `4`. The last token is therefore also a sampled non-target token, so Stage 3D metrics average over multiple final-query identities.

The model uses token-specific representations to compute query/key scores, but the value passed to the classifier is still binary: target evidence or non-target evidence.

- Query/key scores distinguish all token ids `0,1,2,3,4`.
- Attention values remain binary: target maps to `[1,0]`, and every non-target token maps to `[0,1]`.

Therefore, the classifier input is still interpretable as:

```math
(p_t,1-p_t),
```

where $p_t$ is the attention mass on the target token.

## Runs

| Run | Multiplier mode | Max train steps |
|---|---|---:|
| `constant_e50_nt4` | `constant` | 1600 |
| `constant_e100_nt4` | `constant` | 3200 |
| `constant_e1000_nt4` | `constant` | 32000 |
| `log_e50_nt4` | `log` | 1600 |
| `learned_log_e50_nt4` | `learned_log` | 1600 |
| `learned_log_e100_nt4` | `learned_log` | 3200 |
| `learned_log_e200_nt4` | `learned_log` | 6400 |

Run output root:

```text
runs/stage3d_multiple_non_targets/
```

## Overall Results At Length 10M

This table aggregates over sampled last-token types. In other words, examples with different final query types $r$ are pooled together, so the reported margins and denominator statistics should be read as aggregate diagnostics rather than separate per-$r$ measurements.

| Run | Mode | Updates | Positive acc at 10M | Positive logit at 10M | Target attention at 10M | Mean $\Delta_{\min}$ | Worst observed $\Delta_{\min}$ | Mean $c\Delta_{\min}$ | Worst observed $c\Delta_{\min}$ | Type-score std | Generalized attention error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `constant_e50_nt4` | `constant` | 1600 | 0.0 | -3.202 | 0.001010 | 8.806 | 8.195 | n/a | n/a | 0.255 | $1.59\times10^{-8}$ |
| `constant_e100_nt4` | `constant` | 3200 | 0.0 | -4.515 | 0.002505 | 9.728 | 9.151 | n/a | n/a | 0.250 | $7.91\times10^{-8}$ |
| `constant_e1000_nt4` | `constant` | 32000 | 0.0 | -17.223 | 0.053233 | 12.884 | 12.489 | n/a | n/a | 0.226 | $1.20\times10^{-6}$ |
| `log_e50_nt4` | `log` | 1600 | 1.0 | 3.232 | 1.000000 | 4.351 | 3.983 | n/a | n/a | 0.123 | $0.00$ |
| `learned_log_e50_nt4` | `learned_log` | 1600 | 1.0 | 3.002 | 0.958747 | 7.810 | 7.234 | 0.675 | 0.625 | 0.238 | $3.79\times10^{-5}$ |
| `learned_log_e100_nt4` | `learned_log` | 3200 | 1.0 | 4.586 | 0.999770 | 8.293 | 7.758 | 0.968 | 0.905 | 0.231 | $2.19\times10^{-7}$ |
| `learned_log_e200_nt4` | `learned_log` | 6400 | 1.0 | 6.412 | 0.999999 | 8.637 | 8.132 | 1.303 | 1.227 | 0.225 | $9.06\times10^{-8}$ |

For each positive example, $\Delta_{\min}$ is the smallest margin among its non-target token types. Worst observed $\Delta_{\min}$ is the smallest of those values across the positive evaluation examples.

Interpretation:

- Constant multiplier still fails at long length, even after much longer training.
- Fixed log multiplier succeeds because even the worst observed margin is much larger than 1.
- Learned-log e50 and e100 work up to the tested maximum length, 10M, but remain below the simplified asymptotic threshold.
- Learned-log e200 is the only learned-log main run that clearly crosses the threshold under both mean and worst-observed diagnostics.

## Attention Metric Consistency Check

The Stage 3D analysis depends on per-type quantities such as non-target counts, non-target scores, margins, and denominator contributions. These quantities are saved separately from the model's raw attention weights, so I checked whether they are consistent with the actual softmax attention.

For each positive example, I used the recorded per-type counts and scores to reconstruct the target attention mass:

```math
p_t(n\mid r)
=
\frac{1}
{1+\sum_{k=1}^{m}c_k(n)e^{-\alpha\Delta_{r,k}}}.
```

At length 10M, the largest mean absolute error between reconstructed target attention and empirical target attention among the main runs is:

```math
3.79\times10^{-5}.
```

This small error indicates that the recorded per-type metrics are consistent with the model's actual attention behavior, so they can be used for the denominator and margin analysis below.

## Constant Multiplier Behavior

The constant multiplier runs fail at progressively longer lengths as training strength increases:

| Run | First positive failure window | Interpretation |
|---|---|---|
| `constant_e50_nt4` | 1000-10000 | target attention becomes too diluted |
| `constant_e100_nt4` | 10000-100000 | larger margins delay but do not remove failure |
| `constant_e1000_nt4` | 100000-1000000 | very large margins still cannot beat unbounded length growth |

This is the expected behavior. With constant $\alpha$, the denominator contains terms that grow proportionally to sequence length:

```math
\sum_{k=1}^{m}c_k(n)e^{-\alpha\Delta_{r,k}}.
```

Even if every observed $\Delta_{r,k}$ is large, fixed margins do not scale with $\log n$. The model can push the failure point outward, but it does not get an infinite-length solution from constant scaling alone.

## Fixed Log Multiplier Behavior

The fixed log multiplier run succeeds through length 10M.

At length 10M:

- mean $\Delta_{\min}\approx4.351$
- target attention is effectively 1.0
- positive logit remains positive
- positive accuracy is 1.0

This matches the generalized theory. If $\alpha=\log n$, then each denominator term behaves roughly like:

```math
c_k(n)n^{-\Delta_{r,k}}.
```

When all relevant $\Delta_{r,k}>1$, the non-target denominator terms shrink relative to the target term. In this run, even the worst observed non-target margin is well above 1.

## Learned-Log Behavior

The learned-log runs show a training-strength pattern similar to the original Stage 3 result, but with an important nuance: e50 and e100 succeed through 10M even though they do not clearly satisfy the asymptotic condition.

| Run | Length | Positive logit | Target attention | Mean $\Delta_{\min}$ | Mean $c\Delta_{\min}$ |
|---|---:|---:|---:|---:|---:|
| `learned_log_e50_nt4` | 10 | 3.264 | 0.999322 | 7.562 | 0.653 |
| `learned_log_e50_nt4` | 100000 | 3.161 | 0.983331 | 7.579 | 0.655 |
| `learned_log_e50_nt4` | 10000000 | 3.002 | 0.958747 | 7.810 | 0.675 |
| `learned_log_e100_nt4` | 10 | 4.586 | 0.999796 | 8.063 | 0.941 |
| `learned_log_e100_nt4` | 100000 | 4.584 | 0.999641 | 8.079 | 0.943 |
| `learned_log_e100_nt4` | 10000000 | 4.586 | 0.999770 | 8.293 | 0.968 |
| `learned_log_e200_nt4` | 10 | 6.411 | 0.999936 | 8.420 | 1.271 |
| `learned_log_e200_nt4` | 100000 | 6.412 | 0.999995 | 8.435 | 1.273 |
| `learned_log_e200_nt4` | 10000000 | 6.412 | 0.999999 | 8.637 | 1.303 |

Interpretation:

- `learned_log_e50_nt4` is a finite-length success. Its target attention slowly decreases with length, and mean $c\Delta_{\min}<1$.
- `learned_log_e100_nt4` is very close to the threshold. It succeeds through 10M, but both mean and worst-observed $c\Delta_{\min}$ remain below 1 in the observed metrics.
- `learned_log_e200_nt4` crosses the threshold and is the strongest candidate for true asymptotic success in this setup.

The key result is not simply that learned-log succeeds at 10M. The key result is that enough optimization pushes the learned multiplier and worst-case margin into the regime:

```math
c\min_{r,k}\Delta_{r,k}>1.
```

## Does The Model Collapse Non-Target Attention Scores?

No.

If the model collapsed all non-target token types into one shared attention score for the sampled final-query conditions, the non-target type-score standard deviation would be near 0. Instead, the 10M type-score standard deviation is consistently nonzero:

- constant runs: about `0.23` to `0.25`
- learned-log runs: about `0.22` to `0.24`
- fixed-log run: about `0.12`

This means Stage 3D did break the exact two-score assumption at the attention-score level. The model did not simply recreate:

```math
S_n=(a,b,b,\ldots,b).
```

Instead, it learned a multi-score structure conditioned on the last-token type that forms the final query:

```math
S_n^{(r)}=(a_r,b_{r,i_1},b_{r,i_2},\ldots,b_{r,i_{n-1}}).
```

The important result is that length-aware attention can still work without collapsing all non-target token types to the same final-query attention score. This means Stage 3D does not simply reduce back to the original two-score setup.

## Which Non-Target Type Dominates The Denominator?

Aggregated over sampled last-token types, token id `3` is the dominant non-target type in every main run at length 10M.

| Run | Dominant non-target type at 10M | Mean margin | Denominator fraction |
|---|---:|---:|---:|
| `constant_e50_nt4` | 3 | 8.806 | 0.366 |
| `constant_e100_nt4` | 3 | 9.728 | 0.363 |
| `constant_e1000_nt4` | 3 | 12.884 | 0.351 |
| `log_e50_nt4` | 3 | 4.351 | 0.888 |
| `learned_log_e50_nt4` | 3 | 7.810 | 0.526 |
| `learned_log_e100_nt4` | 3 | 8.293 | 0.574 |
| `learned_log_e200_nt4` | 3 | 8.637 | 0.625 |

Token id `3` takes the largest share of the non-target denominator. This is not because it appears much more often since non-target tokens were sampled uniformly. It happens because its score is closest to the target score on average.

I also checked the saved checkpoint weights directly. For each last-token type $r\in\{1,2,3,4\}$, I computed the margins:

```math
\Delta_{r,k}=a_r-b_{r,k}.
```

This weight-level calculation confirmed that token id `3` is the smallest-margin non-target for every last-token type in all seven main runs.

For the strongest learned-log run, the direct margins were:

| Last-token type $r$ | Smallest-margin token | $\Delta_{r,1}$ | $\Delta_{r,2}$ | $\Delta_{r,3}$ | $\Delta_{r,4}$ |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 8.951 | 8.925 | 8.457 | 9.040 |
| 2 | 3 | 8.807 | 9.115 | 8.584 | 9.107 |
| 3 | 3 | 8.455 | 8.613 | 8.132 | 8.655 |
| 4 | 3 | 9.300 | 9.427 | 8.908 | 9.490 |

One alternate-seed check with `learned_log_e200_nt4` also produced token id `3` as the bottleneck for every last-token type. However, this is not enough to explain why token id `3` is consistently the bottleneck.

For the `learned_log_e200_nt4` run:

| Non-target id | Mean count | Mean score | Mean margin | Denominator fraction |
|---:|---:|---:|---:|---:|
| 1 | 2500076.2 | -4.848 | 9.022 | 0.174 |
| 2 | 2500191.8 | -4.965 | 9.139 | 0.112 |
| 3 | 2499746.0 | -4.463 | 8.637 | 0.625 |
| 4 | 2499985.0 | -5.028 | 9.203 | 0.089 |

Even though the non-target token types appear at similar frequencies, the type with the smallest margin contributes most of the non-target denominator. Therefore, the long-length risk is controlled by the non-target type with the smallest margin, not by the average non-target type.

## Scaling Across Non-Target Token Count

Using the main `non_target_token_count=4` experiment as the baseline, I added the minimum scaling subset for `non_target_token_count=2` and `non_target_token_count=8`. The comparison therefore covers:

```text
non_target_token_count = 2, 4, 8
```

The goal was to check whether the Stage 3D mechanism remains stable as the number of non-target token types increases while keeping `train_length=10`.

| Run | Non-target count | Mode | Positive acc at 10M | Positive logit at 10M | Target attention at 10M | Mean $\Delta_{\min}$ | Worst observed $\Delta_{\min}$ | Mean $c\Delta_{\min}$ | Worst observed $c\Delta_{\min}$ | Type-score std |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `constant_e100_nt2` | 2 | `constant` | 0.0 | -4.616 | 0.001416 | 9.496 | 9.303 | n/a | n/a | 0.060 |
| `constant_e100_nt4` | 4 | `constant` | 0.0 | -4.515 | 0.002505 | 9.728 | 9.151 | n/a | n/a | 0.250 |
| `constant_e100_nt8` | 8 | `constant` | 0.0 | -4.507 | 0.001685 | 9.308 | 8.995 | n/a | n/a | 0.265 |
| `log_e50_nt2` | 2 | `log` | 1.0 | 3.352 | 1.000000 | 4.168 | 4.034 | n/a | n/a | 0.050 |
| `log_e50_nt4` | 4 | `log` | 1.0 | 3.232 | 1.000000 | 4.351 | 3.983 | n/a | n/a | 0.123 |
| `log_e50_nt8` | 8 | `log` | 1.0 | 3.271 | 1.000000 | 4.007 | 3.806 | n/a | n/a | 0.175 |
| `learned_log_e200_nt2` | 2 | `learned_log` | 1.0 | 6.505 | 0.999992 | 8.596 | 8.420 | 1.191 | 1.167 | 0.054 |
| `learned_log_e200_nt4` | 4 | `learned_log` | 1.0 | 6.412 | 0.999999 | 8.637 | 8.132 | 1.303 | 1.227 | 0.225 |
| `learned_log_e200_nt8` | 8 | `learned_log` | 1.0 | 6.422 | 0.999999 | 8.208 | 7.932 | 1.300 | 1.256 | 0.240 |

The qualitative behavior is stable across `non_target_token_count = 2, 4, 8`.

Constant multiplier fails at length 10M for all three vocabulary sizes. In all cases, the first positive failure occurs between length 10000 and 100000. This confirms that increasing the raw margin can delay failure, but constant attention scaling still cannot defeat unbounded length growth.

Fixed log multiplier succeeds for all three vocabulary sizes. The worst observed margin remains well above 1:

```text
nt2: worst Delta_min = 4.034
nt4: worst Delta_min = 3.983
nt8: worst Delta_min = 3.806
```

Learned-log e200 also succeeds for all three vocabulary sizes. More importantly, the worst observed $c\Delta_{\min}$ remains above 1:

```text
nt2: worst cDelta_min = 1.167
nt4: worst cDelta_min = 1.227
nt8: worst cDelta_min = 1.256
```

As the number of non-target token types increases, the worst observed margin becomes slightly smaller:

```text
nt2: worst Delta_min = 8.420
nt4: worst Delta_min = 8.132
nt8: worst Delta_min = 7.932
```

This is expected. $\Delta_{\min}$ is a minimum over more non-target token types, so adding more non-target types increases the chance that one type becomes a smaller-margin competitor.

However, the learned coefficient $c$ increases at the same time:

```text
nt2: c = 0.1386
nt4: c = 0.1509
nt8: c = 0.1584
```

As a result, the effective worst-case quantity $c\Delta_{\min}$ stays above 1. In these runs, learned-log attention appears to compensate for the smaller-margin non-target set by increasing the log-length multiplier.

The non-target type-score standard deviation also increases as more non-target token types are added:

```text
learned_log_e200_nt2: type-score std = 0.054
learned_log_e200_nt4: type-score std = 0.225
learned_log_e200_nt8: type-score std = 0.240
```

This means the model does not collapse all non-target token types into one shared attention score. Instead, it learns distinct non-target scores, and the generalized worst-margin analysis remains necessary.

The denominator-dominant non-target type changes with vocabulary size:

| Run | Dominant non-target token | Margin | Denominator fraction |
|---|---:|---:|---:|
| `learned_log_e200_nt2` | 2 | 8.596 | 0.584 |
| `learned_log_e200_nt4` | 3 | 8.637 | 0.625 |
| `learned_log_e200_nt8` | 5 | 8.208 | 0.362 |

Within the range `non_target_token_count = 2, 4, 8`, the Stage 3D conclusion remains stable. The model does not rely on exact non-target score collapse. Constant scaling still fails, fixed-log scaling succeeds, and learned-log e200 succeeds with worst observed $c\Delta_{\min}>1$.

Because `train_length=10` gives only 9 non-target positions in each positive example, increasing `non_target_token_count` beyond 9 would test a different regime: non-target vocabulary diversity larger than per-example context capacity.

## Main Conclusion

**Stage 3D shows that exact non-target attention-score collapse is not necessary in the reduced model.**

The model does not force all non-target token types to share one attention score. Instead, it learns distinct non-target scores, and long-length behavior is governed by the final-query/non-target-key pair with the smallest margin:

```math
\Delta_{\min}=\min_{r,k}(a_r-b_{r,k}).
```

The strongest learned-log run reaches:

```math
\mathrm{mean}\ c\Delta_{\min}\approx1.303>1,
\qquad
\mathrm{worst\ observed}\ c\Delta_{\min}\approx1.227>1,
```

which is consistent with the generalized asymptotic condition.

The scaling checks show the same qualitative pattern for `non_target_token_count = 2, 4, 8`: constant scaling fails, fixed-log scaling succeeds, and learned-log e200 succeeds with worst observed $c\Delta_{\min}>1$.

Across vocabulary sizes, the dominant token id changes, so the denominator bottleneck should be interpreted through the smallest margin.

Therefore, the Stage 3 mechanism extends beyond the exact two-score assumption. In this reduced setup, length-aware attention can generalize across multiple non-target token types as long as the learned multiplier and the worst-case margin are large enough.

## Limitations

- This report analyzes `non_target_token_count = 2, 4, 8`, but it does not test values above the training-context non-target capacity.
- Non-target sampling is uniform; skewed non-target distributions are a future follow-up.
- The target is fixed at position 0; combining Stage 3C target-anywhere with Stage 3D multiple non-target types is a future follow-up.
- The last token is a sampled non-target token, so the current metrics are averaged over different final query identities.
- The report does not yet include a per-final-query-type breakdown, so aggregate bottleneck claims should not be read as proving the same bottleneck for every $r$.
- Attention values are still binary and fixed; distinct or learned non-target value vectors are not tested here.
- This result applies to the reduced Stage 3 model and does not automatically transfer to full transformers. Full transformers include learned value vectors, residual streams, MLPs, layer normalization, multiple heads, and classifier or pooling effects, so this reduced-model mechanism may not transfer directly.
