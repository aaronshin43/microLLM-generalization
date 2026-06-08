# Stage 3 Simplified Attention Empirical Analysis

## Objective

This document analyzes the Stage 3 simplified attention experiment.

The goal was to test whether the theoretical model in `SIMPLIFIED_LENGTH_AWARE_ATTENTION_MODEL.md` matches empirical behavior when the exact simplified architecture is trained.

The core theoretical prediction is:

```math
p_t(n)
=
\frac{e^{\alpha(n)\Delta}}
{e^{\alpha(n)\Delta} + (n-1)}.
```

Variables used here:

- $p_t(n)$ is the target attention mass at sequence length $n$.
- $\alpha(n)$ is the length-dependent score multiplier.
- $\Delta = a-b$ is the target-vs-non-target score margin.
- $a$ is the target key score.
- $b$ is the non-target key score.

The empirical question was:

```text
If the model is trained rather than manually assigned scores, does its measured target attention follow this formula?
```

## Runs Analyzed

Analyzed run directories:

```text
runs/stage3_simplified_attention_constant
runs/stage3_simplified_attention_log
runs/stage3_simplified_attention_learned_log
```

All three runs used:

- target token id: $t=0$
- non-target token id: $u=1$
- fixed one-hot values: $t \mapsto [1,0]$, $u \mapsto [0,1]$
- training length: 10
- training examples: 2000
- epochs: 50
- evaluation lengths: 10 through 10000

## Model Summary

The model is a minimal last-query attention classifier.

For a positive sequence:

```text
t, u, u, ..., u
```

the model computes one last-query score row:

```math
S_n = (a,b,b,\ldots,b).
```

Then it applies a length-dependent scale:

```math
\tilde{s}_j = \alpha(n)s_j.
```

The attention output is:

```math
o(n)
=
(p_{\mathrm{emp}}(n), 1-p_{\mathrm{emp}}(n)).
```

Variables used here:

- $p_{\mathrm{emp}}(n)$ is the measured target attention mass.
- the first coordinate is target mass because $t \mapsto [1,0]$.
- the second coordinate is non-target mass because $u \mapsto [0,1]$.

The final classifier is:

```math
z(n)
=
w^\top o(n) + b_{\mathrm{cls}}.
```

Positive examples are classified correctly when $z(n) \ge 0$.

## High-Level Finding

The simplified theory matches the empirical attention behavior almost exactly.

Across all three runs:

- non-target score standard deviation was exactly 0.0
- measured $\Delta$ was constant across sequence length
- empirical target attention matched theoretical target attention up to numerical precision
- the constant-$\alpha$ run failed exactly when target attention fell below the classifier's effective threshold

Summary:

```text
When the architecture enforces the simplified assumptions, the theoretical formula describes practice extremely well.
```

## Overall Results

| Run | $\Delta$ | $\alpha(n)$ behavior | $p_{\mathrm{emp}}(10)$ | $p_{\mathrm{emp}}(10000)$ | Positive logit at 10000 | Accuracy at 10000 |
|---|---:|---|---:|---:|---:|---:|
| `constant` | 8.1006 | $1$ | 0.9973 | 0.2479 | -1.8080 | 0.5000 |
| `log` | 3.9361 | $\log n$ | 0.9990 | 1.0000 | 3.7196 | 1.0000 |
| `learned_log` | 7.4085 | $1+0.0661\log(1+n)$ | 0.9983 | 0.9373 | 3.3152 | 1.0000 |

Interpretation:

- Constant $\alpha$ learns a large fixed margin, but fixed margin eventually loses to the growing denominator.
- $\alpha(n)=\log n$ learns a smaller raw margin, but because $\Delta>1$, the theory predicts target attention should approach 1.
- Learned log scaling learns a weak coefficient, but the raw margin is large enough to succeed through length 10000.

## Theory-Vs-Empirical Match

| Run | Max attention error | Non-target score std | $\Delta$ stable by length? | Theory match? |
|---|---:|---:|---|---|
| `constant` | about $6.6 \times 10^{-7}$ | 0.0 | yes | yes |
| `log` | about $1.2 \times 10^{-7}$ | 0.0 | yes | yes |
| `learned_log` | about $1.6 \times 10^{-6}$ | 0.0 | yes | yes |

The attention error is:

```math
\left|
p_{\mathrm{emp}}(n)
-
p_{\mathrm{theory}}(n)
\right|.
```

The error is essentially numerical floating-point noise.

Interpretation:

The model does not merely follow the same qualitative trend as the theory. It realizes the exact two-score structure assumed by the theory.

## Constant Alpha Run

The constant run learned:

```math
\Delta \approx 8.1006.
```

Since:

```math
\alpha(n)=1,
```

the target attention mass is:

```math
p_t(n)
=
\frac{e^\Delta}
{e^\Delta + (n-1)}.
```

This must decay as $n$ grows.

Observed target attention:

| Length | Empirical target attention | Positive logit | Accuracy |
|---:|---:|---:|---:|
| 10 | 0.9973 | 3.7672 | 1.0000 |
| 1000 | 0.7674 | 2.0571 | 1.0000 |
| 2000 | 0.6225 | 0.9788 | 1.0000 |
| 5000 | 0.3974 | -0.6961 | 0.5000 |
| 10000 | 0.2479 | -1.8080 | 0.5000 |

The learned classifier's approximate positive-decision threshold in terms of target attention mass was:

```math
p_t(n) \approx 0.4909.
```

So the failure at length 5000 is expected: by then, target attention has fallen below the classifier threshold.

Summary:

```text
The constant-alpha run is a clean empirical example of fixed-margin attention dilution.
```

## Log Alpha Run

The log run learned:

```math
\Delta \approx 3.9361.
```

With:

```math
\alpha(n)=\log n,
```

the theory gives:

```math
p_t(n)
=
\frac{n^\Delta}
{n^\Delta+n-1}.
```

Since:

```math
\Delta > 1,
```

the target attention mass should approach 1.

Observed behavior:

- target attention was already 0.9990 at length 10
- target attention rounded to 1.0000 by length 500
- positive logits stayed around 3.72 through length 10000
- accuracy stayed 1.0000 at every evaluated length

Summary:

```text
The log-alpha run confirms the theoretical threshold: when alpha is log n and Delta > 1, target attention remains dominant.
```

## Learned Log Run

The learned-log run used:

```math
\alpha(n)
=
1+\mathrm{softplus}(k_\alpha)\log(1+n).
```

It learned:

```math
\mathrm{softplus}(k_\alpha)
\approx
0.0661.
```

The raw margin was:

```math
\Delta \approx 7.4085.
```

Therefore the asymptotic log coefficient is approximately:

```math
0.0661 \times 7.4085
\approx
0.4894.
```

This is less than 1.

Interpretation:

The learned-log model succeeds through length 10000, but the asymptotic condition is not satisfied.

At length 10000:

```math
p_{\mathrm{emp}}(10000)
\approx
0.9373.
```

This is still far above the classifier threshold, so accuracy remains perfect. However, because the effective asymptotic exponent is below 1, the theory predicts eventual decay at sufficiently larger lengths.

A rough attention-level crossover estimate is around:

```math
n \approx 2.0 \times 10^6.
```

Summary:

```text
Learned log scaling worked at the tested lengths, but it did not learn a strong enough asymptotic correction to guarantee infinite-length success.
```

## Why The Theory Matches So Well

The match is strong because the implementation makes the simplified assumptions true.

The positive input has one target token and many identical non-target tokens:

```text
t, u, u, ..., u
```

The embeddings are fixed one-hot vectors:

```math
t \mapsto [1,0],
\qquad
u \mapsto [0,1].
```

All non-target tokens are identical. Therefore their key vectors are identical, and their attention scores are identical:

```math
S_n = (a,b,b,\ldots,b).
```

That is why:

```math
\operatorname{std}_{j \ne t}(s_j)=0.
```

Once this condition holds, the theoretical formula is exactly the softmax formula, not an approximation.

## Connection To Stage 2B

This Stage 3 result clarifies what Stage 2B was trying to do.

Stage 2B modifies attention scores so the target numerator can compete with a denominator that grows with length.

In the simplified model, the condition is:

```math
\alpha(n)\Delta - \log n \to +\infty.
```

Stage 3 shows this condition is empirically correct when the model actually satisfies the simplified assumptions.

However, Stage 2B is harder because the full transformer may violate those assumptions:

- non-target scores may not all be equal
- $\Delta$ may vary across examples or lengths
- max pooling can introduce additional non-target interference
- the classifier may depend on features beyond target attention mass
- learned length corrections may be too weak even if they help at finite lengths

Interpretation:

Stage 3 validates the attention-dilution theory in a controlled setting. Stage 2B remains harder because it must make a real transformer approximate the same favorable score geometry.

## Answer To The Main Question

Question:

```text
Does the theory match the practice when this exact model is used for the learning process?
```

Answer:

```text
Yes, for attention behavior, almost exactly.
```

The trained simplified model produces the assumed score structure, and the measured attention mass follows the theoretical formula almost perfectly.

For classification behavior, the theory also explains the observed success or failure once the learned classifier threshold is considered.

The most important caveat is the learned-log run:

```text
Finite-length success does not imply infinite-length success.
```

Even though it succeeds through length 10000, its learned coefficient is too small to satisfy the asymptotic condition.

## Next Steps

Recommended follow-up experiments:

- Evaluate the learned-log run at lengths beyond 10000, especially near $10^6$ to $10^7$.
- Add a fixed $c\log n$ mode with controlled $c\Delta<1$, $c\Delta=1$, and $c\Delta>1$ regimes.
- Repeat the experiment with learned embeddings to test whether the two-score structure still emerges.
- Move the target position away from index 0 to test whether the last-query setup still produces the same geometry.
- Compare this simplified result directly against Stage 2B target-key log-bias attention summaries.
