# Simplified Length-Aware Attention Model

## Objective

This document derives a minimal two-token model for length-aware attention and connects it to the Stage 1 and Stage 2B results.

The goal is not to model every component of the trained transformer. The goal is to isolate the softmax denominator effect and state exactly when a length-aware correction can preserve target attention as sequence length grows.

## Simplified Setup

Use a vocabulary with two tokens:

- $t$: target token
- $u$: non-target token

For sequence length $n$, assume the input is:

```text
t, u, u, ..., u
```

Use embeddings:

```math
t \mapsto [1, 0],
\qquad
u \mapsto [0, 1].
```

The embedded input matrix is:

```math
X_n =
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
0 & 1 \\
\vdots & \vdots \\
0 & 1
\end{bmatrix}.
```

Assume the final classifier reads only the last token output. For the last query row, assume the score vector is:

```math
S_n = (a, b, b, \ldots, b),
\qquad
a > b.
```

Define the fixed target-vs-non-target score margin:

```math
\Delta = a - b > 0.
```

Apply inverse temperature $\alpha$ before softmax:

```math
\operatorname{softmax}(\alpha S_n).
```

## Target Attention Mass

The target attention mass is:

```math
p_t(n)
=
\frac{e^{\alpha a}}
{e^{\alpha a} + (n-1)e^{\alpha b}}.
```

Dividing the numerator and denominator by $e^{\alpha b}$ gives:

```math
p_t(n)
=
\frac{e^{\alpha(a-b)}}
{e^{\alpha(a-b)} + (n-1)}
=
\frac{e^{\alpha\Delta}}
{e^{\alpha\Delta} + (n-1)}.
```

The attention output after multiplying by $X_n$ is:

```math
(p_t(n), 1 - p_t(n)).
```

This output cannot converge to $(1, 1)$. It is a convex combination of $[1, 0]$ and $[0, 1]$, so its coordinates always sum to 1.

## Constant Temperature

If:

```math
\alpha = \alpha_0,
```

then:

```math
p_t(n)
=
\frac{e^{\alpha_0\Delta}}
{e^{\alpha_0\Delta} + (n-1)}
\to 0
\qquad
\text{as } n \to \infty.
```

Interpretation:

A fixed margin can beat each individual non-target token, but it cannot beat the growing number of non-target tokens in the softmax denominator.

## Log Temperature

If:

```math
\alpha = \log n,
```

then:

```math
p_t(n)
=
\frac{n^\Delta}
{n^\Delta + n - 1}.
```

Therefore:

```math
\Delta > 1 \Rightarrow p_t(n) \to 1,
\qquad
\Delta = 1 \Rightarrow p_t(n) \to \frac{1}{2},
\qquad
0 < \Delta < 1 \Rightarrow p_t(n) \to 0.
```

Interpretation:

$\log n$ is not automatically sufficient. It works only when the target score margin is large enough.

## Scaled Log Temperature

If:

```math
\alpha = c\log n,
```

then:

```math
p_t(n)
=
\frac{n^{c\Delta}}
{n^{c\Delta} + n - 1}.
```

Therefore:

```math
c\Delta > 1 \Rightarrow p_t(n) \to 1,
\qquad
c\Delta = 1 \Rightarrow p_t(n) \to \frac{1}{2},
\qquad
c\Delta < 1 \Rightarrow p_t(n) \to 0.
```

Interpretation:

The learned coefficient matters. A global log-temperature intervention can still fail if the effective product $c\Delta$ is below the threshold.

## Visualization

The plot below compares the simplified target attention mass across the constant, log, and scaled-log regimes.

<img src="figures/simplified_length_aware_attention_pt.png" alt="Simplified length-aware attention regimes" width="560">

## General Condition

Start from the target attention mass:

```math
p_t(n)
=
\frac{e^{\alpha\Delta}}
{e^{\alpha\Delta} + (n-1)}.
```

Divide the numerator and denominator by $e^{\alpha\Delta}$:

```math
p_t(n)
=
\frac{1}
{1 + (n-1)e^{-\alpha\Delta}}.
```

Define the non-target-to-target softmax ratio:

```math
R_n
=
(n-1)e^{-\alpha\Delta}.
```

Then:

```math
p_t(n)
=
\frac{1}{1+R_n}.
```

This form makes the limit condition explicit. Target attention converges to 1 exactly when the non-target-to-target ratio goes to 0:

```math
R_n \to 0.
```

Taking the log of $R_n$ gives:

```math
\log R_n
=
\log(n-1) - \alpha\Delta.
```

For large $n$, $\log(n-1)$ has the same asymptotic growth as $\log n$. Therefore:

```math
R_n \to 0
```

when:

```math
\alpha\Delta - \log n \to +\infty.
```

Target attention converges to 0 when the opposite happens:

```math
\alpha\Delta - \log n \to -\infty.
```

There is also a boundary case. If:

```math
\alpha\Delta - \log n \to C,
```

where $C$ is finite, then:

```math
\log R_n \to -C,
\qquad
R_n \to e^{-C},
```

so:

```math
p_t(n)
\to
\frac{1}{1+e^{-C}}.
```

Interpretation:

The scaled target margin must grow faster than the logarithm of the number of competing non-target keys. The condition comes from controlling the non-target-to-target ratio in the softmax denominator.

## Connection To Stage 1

Stage 1 used a standard transformer trained at length 10. The empirical analysis showed that target attention decreased with length and exactly-one positive logits eventually crossed below zero.

The simplified model explains this pattern:

```math
p_t(n)
=
\frac{e^{\mathrm{fixed\ target\ margin}}}
{e^{\mathrm{fixed\ target\ margin}} + (n-1)}.
```

Even if the target key has a higher score than each non-target key, the total non-target mass grows with length. The model can therefore learn a locally correct detector at length 10 without learning a length-invariant attention rule.

## Connection To Stage 2B

### Global Log-Temperature

The global log-temperature intervention scales all attention scores:

```math
\tilde{s}_{ij} = \alpha s_{ij}.
```

Here, $s_{ij}$ is the raw attention score from query position $i$ to key position $j$; in the simplified last-query case, that row is $S_n = (a,b,b,\ldots,b)$.

In the simplified model, this succeeds only when the scaled margin satisfies:

```math
\alpha\Delta - \log n \to +\infty.
```

For a $c\log n$ schedule, this requires:

```math
c\Delta > 1.
```

Interpretation:

Global temperature can fail if the learned coefficient is too weak or if the model's underlying target margin is too small. This matches the Stage 2B result where global log-temperature did not produce robust long-length generalization.

### Target-Key Log-Bias

The target-key bias intervention adds a key-specific term:

```math
\tilde{s}_{ij}
=
s_{ij}
+
\beta(n)r_j.
```

Here, $s_{ij}$ is the raw attention score, and $r_j$ is a learned target-likeness score for key position $j$.

This changes the margin directly:

```math
\Delta_{\mathrm{eff}}(n)
=
\Delta_{\mathrm{base}}
+
\beta(n)(r_t-r_u).
```

Interpretation:

This intervention is better aligned with the task because the problem is not just that all scores are too small. The problem is that the target score must remain dominant over a growing number of non-target keys.

## Finite-Length Success Is Not An Infinite Guarantee

The target-key log-bias multi-length run succeeded through length 10000, but the simplified model warns against overclaiming.

Finite success does not prove infinite generalization if:

- target attention still decreases with length
- positive logits still decrease with length
- attention entropy still increases with length
- the effective margin condition may eventually fail

The key empirical question is whether the learned effective margin keeps satisfying:

```math
\Delta_{\mathrm{eff}}(n) - \log n \to +\infty.
```

If not, the model may only have pushed the failure point farther out.
