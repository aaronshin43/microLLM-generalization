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

The important margin is no longer the average margin. The bottleneck is the smallest margin:

```math
\Delta_{\min}=\min_{r,k}\Delta_{r,k}.
```

The non-target type with the smallest margin has the largest score among the non-target types for some final-query condition, so its denominator term decays slowest. If that token type appears with frequency proportional to $n$, it can dominate long-length behavior.

The implementation records per-example margins and aggregates them across sampled last-token types. Therefore, the reported $\Delta_{\min}$ values below are empirical aggregate diagnostics, not a full per-$r$ exhaustive table.

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

The score representation and attention value representation are separated:

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

This table aggregates over sampled last-token types.

| Run | Mode | Updates | Positive acc at 10M | Positive logit at 10M | Target attention at 10M | Mean $\Delta_{\min}$ | Worst observed $\Delta_{\min}$ | Mean $c\Delta_{\min}$ | Worst observed $c\Delta_{\min}$ | Type-score std | Generalized attention error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `constant_e50_nt4` | `constant` | 1600 | 0.0 | -3.202 | 0.001010 | 8.806 | 8.195 | n/a | n/a | 0.255 | $1.59\times10^{-8}$ |
| `constant_e100_nt4` | `constant` | 3200 | 0.0 | -4.515 | 0.002505 | 9.728 | 9.151 | n/a | n/a | 0.250 | $7.91\times10^{-8}$ |
| `constant_e1000_nt4` | `constant` | 32000 | 0.0 | -17.223 | 0.053233 | 12.884 | 12.489 | n/a | n/a | 0.226 | $1.20\times10^{-6}$ |
| `log_e50_nt4` | `log` | 1600 | 1.0 | 3.232 | 1.000000 | 4.351 | 3.983 | n/a | n/a | 0.123 | $0.00$ |
| `learned_log_e50_nt4` | `learned_log` | 1600 | 1.0 | 3.002 | 0.958747 | 7.810 | 7.234 | 0.675 | 0.625 | 0.238 | $3.79\times10^{-5}$ |
| `learned_log_e100_nt4` | `learned_log` | 3200 | 1.0 | 4.586 | 0.999770 | 8.293 | 7.758 | 0.968 | 0.905 | 0.231 | $2.19\times10^{-7}$ |
| `learned_log_e200_nt4` | `learned_log` | 6400 | 1.0 | 6.412 | 0.999999 | 8.637 | 8.132 | 1.303 | 1.227 | 0.225 | $9.06\times10^{-8}$ |

Interpretation:

- Constant multiplier still fails at long length, even after much longer training.
- Fixed log multiplier succeeds because even the worst observed margin is much larger than 1.
- Learned-log e200 reaches $c\Delta_{\min}>1$ using both the mean and worst-observed aggregate diagnostics.
- Learned-log e50 and e100 succeed through 10M, but both their mean and worst-observed $c\Delta_{\min}$ values are below 1, so they should be interpreted as finite-length successes rather than confirmed asymptotic solutions.

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

The important result is that length-aware attention can still work in this reduced model without exact non-target attention-score collapse. This is a score-level claim; it does not by itself prove that the non-target key vectors are geometrically distinct in every possible sense.

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

Because non-target tokens were sampled uniformly, each non-target type appears about equally often. At length 10M, each type appears about 2.5M times on average. Therefore, token id `3` dominates the aggregate denominator mainly because it has the smallest aggregate margin, not because it appears much more often.

This does not prove that token id `3` is the bottleneck for every final-query token type $r$. It means token id `3` is the dominant bottleneck after averaging over the sampled last-token types in the current metrics.

For the strongest learned-log run:

| Non-target id | Mean count | Mean score | Mean margin | Denominator fraction |
|---:|---:|---:|---:|---:|
| 1 | 2500076.2 | -4.848 | 9.022 | 0.174 |
| 2 | 2500191.8 | -4.965 | 9.139 | 0.112 |
| 3 | 2499746.0 | -4.463 | 8.637 | 0.625 |
| 4 | 2499985.0 | -5.028 | 9.203 | 0.089 |

This supports the worst-case-margin view: long-length behavior is controlled more by the hardest non-target type than by the mean non-target margin.

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

## Theory-Practice Attention Match

The generalized theory prediction matches empirical target attention with very small error. At length 10M, the largest mean absolute error among the main runs is about:

```math
1.20\times10^{-6}.
```

This should not be interpreted as surprising evidence by itself. The generalized formula is the full softmax rewritten using the recorded per-type counts and per-type scores. Once those counts and scores are measured correctly, the formula should match.

The meaningful empirical result is that the model produces non-identical non-target attention scores, and the observed behavior is still explained by the generalized denominator and the smallest margin.

## Answers To The Main Questions

### 1. Does The Model Collapse All Non-Target Token Types Into One Shared Attention Score?

No.

The non-target type-score standard deviation is consistently nonzero. The model keeps distinct non-target attention scores in the sampled final-query conditions.

### 2. If Non-Target Scores Differ, Does The Worst-Case Margin Become Large Enough?

Yes for fixed-log and learned-log e200.

The fixed-log run has worst observed $\Delta_{\min}>1$ by a large margin. The learned-log e200 run has:

```math
\mathrm{mean}\ c\Delta_{\min}\approx1.303>1,
\qquad
\mathrm{worst\ observed}\ c\Delta_{\min}\approx1.227>1.
```

The learned-log e50 and e100 runs do not clearly satisfy the asymptotic condition, despite succeeding through 10M.

### 3. Is Long-Length Success Controlled By $c\min_{r,k}\Delta_{r,k}>1$?

The results support this interpretation.

For learned-log attention, e200 is the run that clearly crosses the threshold under both mean and worst-observed aggregate metrics. The earlier learned-log runs succeed on the finite sweep but remain theoretically weaker because their $c\Delta_{\min}$ values are below 1.

### 4. Which Non-Target Token Type Contributes Most To The Denominator?

Token id `3`, after aggregation over sampled last-token types.

It consistently has the smallest margin and the largest denominator fraction. This confirms that the hardest non-target type is the correct object to inspect.

### 5. Do Constant, Log, And Learned-Log Behave Differently?

Yes.

Constant multiplier fails at long length. Fixed log multiplier succeeds because all observed margins are above the threshold. Learned-log can reach the asymptotic regime, but only with enough optimization.

## Main Conclusion

**Stage 3D shows that exact non-target attention-score collapse is not necessary in the reduced model.**

The model does not force all non-target token types to share one attention score. Instead, it learns distinct non-target scores, and long-length behavior is governed by the hardest final-query/non-target-key pair:

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

Therefore, the Stage 3 mechanism extends beyond the exact two-score assumption. In this reduced setup, length-aware attention can generalize with multiple non-target token types as long as the learned multiplier and the worst-case margin are large enough.

## Limitations

- This report only analyzes the main `non_target_token_count=4` conditions.
- The scaling conditions for `non_target_token_count=2` and `non_target_token_count=8` have not been analyzed here.
- Non-target sampling is uniform; skewed non-target distributions are a later follow-up.
- The target is fixed at position 0; combining Stage 3C target-anywhere with Stage 3D multiple non-target types is a later follow-up.
- The last token is a sampled non-target token, so the current metrics are averaged over different final query identities.
- The report does not yet include a per-final-query-type breakdown, so aggregate bottleneck claims should not be read as proving the same bottleneck for every $r$.
- Attention values are still binary and fixed; distinct or learned non-target value vectors are not tested here.
- This result applies to the reduced Stage 3 model and does not automatically transfer to full transformers.
