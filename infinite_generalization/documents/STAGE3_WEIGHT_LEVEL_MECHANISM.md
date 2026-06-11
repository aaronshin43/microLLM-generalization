# Stage 3 Weight-Level Mechanism

## Objective

This document gives a detailed weight-level explanation of how the simplest Stage 3 model creates the target-vs-non-target score margin:

```math
\Delta=a-b.
```

This margin is the quantity that enters the simplified target-attention formula:

```math
p_t(n)
=
\frac{e^{\alpha\Delta}}
{e^{\alpha\Delta}+(n-1)}.
```

Here:

- $p_t(n)$ is the target attention mass at sequence length $n$.
- $\alpha$ is the score multiplier used at that length.
- $n-1$ is the number of non-target keys competing with the target key.
- Larger $\Delta$ increases the target numerator $e^{\alpha\Delta}$ relative to the non-target denominator term.

It is a detailed version of the `Weight-Level Mechanism` section in `STAGE3_SIMPLIFIED_LENGTH_AWARE_ATTENTION.md`.

The main example is the rerun:

```text
runs/stage3_mechanistic_interpretation/learned_log_e200
```

## Setup

The model uses two fixed one-hot token vectors:

```math
x_t=
\begin{bmatrix}
1 \\
0
\end{bmatrix},
\qquad
x_u=
\begin{bmatrix}
0 \\
1
\end{bmatrix}.
```

Here:

- $t$ is the target token.
- $u$ is the non-target token.
- positive inputs have the form $t,u,u,\ldots,u$.
- the final token is always $u$, so the final query is $q_u$.

The model has learned query and key projections:

```math
W_Q \in \mathbb{R}^{2\times2},
\qquad
W_K \in \mathbb{R}^{2\times2}.
```

PyTorch stores `nn.Linear(2, d_head, bias=False)` weights with shape `[d_head, input_dim]`. Since $d_{\mathrm{head}}=2$, both matrices are $2\times2$.

## Learned Weights For `learned_log_e200`

The learned query matrix is:

```math
W_Q
=
\begin{bmatrix}
0.5406 & 2.1707 \\
-0.1657 & 1.8260
\end{bmatrix}.
```

The learned key matrix is:

```math
W_K
=
\begin{bmatrix}
1.5457 & -1.5580 \\
1.4076 & -1.3366
\end{bmatrix}.
```

Because $x_t=[1,0]^\top$ and $x_u=[0,1]^\top$, multiplying by a one-hot vector simply selects a column.

Therefore:

```math
q_t=W_Qx_t
=
\begin{bmatrix}
0.5406 \\
-0.1657
\end{bmatrix},
\qquad
q_u=W_Qx_u
=
\begin{bmatrix}
2.1707 \\
1.8260
\end{bmatrix}.
```

Similarly:

```math
k_t=W_Kx_t
=
\begin{bmatrix}
1.5457 \\
1.4076
\end{bmatrix},
\qquad
k_u=W_Kx_u
=
\begin{bmatrix}
-1.5580 \\
-1.3366
\end{bmatrix}.
```

The final query is $q_u$ because the last input token is $u$.

## Computing The Target Score $a$

The target key score is:

```math
a
=
\frac{q_u^\top k_t}{\sqrt{d}}.
```

Here $d=2$, so:

```math
\sqrt{d}=\sqrt{2}\approx1.4142.
```

First compute the dot product:

```math
q_u^\top k_t
=
(2.1707)(1.5457)+(1.8260)(1.4076).
```

Component-wise:

```math
(2.1707)(1.5457)\approx3.3552,
```

```math
(1.8260)(1.4076)\approx2.5704.
```

So:

```math
q_u^\top k_t
\approx
3.3552+2.5704
=
5.9256.
```

After scaling:

```math
a
\approx
\frac{5.9256}{1.4142}
=
4.1900.
```

## Computing The Non-Target Score $b$

The non-target key score is:

```math
b
=
\frac{q_u^\top k_u}{\sqrt{d}}.
```

First compute the dot product:

```math
q_u^\top k_u
=
(2.1707)(-1.5580)+(1.8260)(-1.3366).
```

Component-wise:

```math
(2.1707)(-1.5580)\approx-3.3818,
```

```math
(1.8260)(-1.3366)\approx-2.4407.
```

So:

```math
q_u^\top k_u
\approx
-3.3818-2.4407
=
-5.8225.
```

After scaling:

```math
b
\approx
\frac{-5.8225}{1.4142}
=
-4.1171.
```

## Computing The Margin $\Delta$

The margin is:

```math
\Delta=a-b.
```

Using the scores above:

```math
\Delta
\approx
4.1900-(-4.1171)
=
8.3071.
```

The saved analysis gives:

```math
\Delta=8.3072.
```

This exactly matches the recorded `mean_delta` in `metrics_by_length.csv`.

## Equivalent Difference-Vector Calculation

The same margin can be computed more directly:

```math
\Delta
=
\frac{q_u^\top(k_t-k_u)}{\sqrt{d}}.
```

First compute:

```math
k_t-k_u
=
\begin{bmatrix}
1.5457-(-1.5580) \\
1.4076-(-1.3366)
\end{bmatrix}
=
\begin{bmatrix}
3.1037 \\
2.7442
\end{bmatrix}.
```

Now compute the component-wise product:

```math
q_u\odot(k_t-k_u)
=
\begin{bmatrix}
(2.1707)(3.1037) \\
(1.8260)(2.7442)
\end{bmatrix}
=
\begin{bmatrix}
6.7371 \\
5.0110
\end{bmatrix}.
```

Sum the components:

```math
q_u^\top(k_t-k_u)
\approx
6.7371+5.0110
=
11.7481.
```

Scale by $\sqrt{2}$:

```math
\Delta
\approx
\frac{11.7481}{1.4142}
=
8.3072.
```

This is the same value as $a-b$.

## Per-Dimension Margin Contributions

Each dimension's contribution to $\Delta$ is:

```math
\frac{q_{u,i}(k_{t,i}-k_{u,i})}{\sqrt{d}}.
```

For `learned_log_e200`:

| Dimension | $q_u$ | $k_t-k_u$ | raw product | contribution to $\Delta$ |
|---:|---:|---:|---:|---:|
| 0 | 2.1707 | 3.1037 | 6.7371 | 4.7638 |
| 1 | 1.8260 | 2.7442 | 5.0110 | 3.5433 |
| total | n/a | n/a | 11.7481 | 8.3072 |

Both dimensions contribute positively. This means the final non-target query $q_u$ is aligned with the target-minus-non-target key direction $k_t-k_u$ in both hidden dimensions.

## Interpretation

The mechanism is simple:

```math
q_u^\top k_t > q_u^\top k_u.
```

Equivalently:

```math
q_u^\top(k_t-k_u)>0.
```

The learned $W_Q$ makes the final non-target query $q_u$ point in a positive direction. The learned $W_K$ maps the target token to a positive key vector and the non-target token to a negative key vector. Therefore, the same query $q_u$ has a positive dot product with $k_t$ and a negative dot product with $k_u$.

This creates:

```math
a\approx4.1900,
\qquad
b\approx-4.1171,
\qquad
\Delta\approx8.3072.
```

That positive margin is the raw score advantage that later enters the length-aware attention formula.

## Comparison Across Stage 3 Runs

The same direct calculation was run for all rerun Stage 3 checkpoints.

| Run | Mode | Updates | $a$ | $b$ | $\Delta$ | Dim 0 contribution | Dim 1 contribution |
|---|---|---:|---:|---:|---:|---:|---:|
| `constant_e50` | constant | 1600 | 4.0935 | -4.0222 | 8.1156 | 4.7018 | 3.4139 |
| `constant_e100` | constant | 3200 | 4.5421 | -4.4655 | 9.0076 | 5.0958 | 3.9118 |
| `constant_e1000` | constant | 32000 | 6.1961 | -6.1013 | 12.2974 | 6.4676 | 5.8298 |
| `log_e50` | log | 1600 | 1.9971 | -1.9458 | 3.9429 | 2.3800 | 1.5629 |
| `learned_log_e50` | learned_log | 1600 | 3.7427 | -3.6753 | 7.4180 | 4.3660 | 3.0520 |
| `learned_log_e100` | learned_log | 3200 | 4.0053 | -3.9347 | 7.9400 | 4.6012 | 3.3388 |
| `learned_log_e200` | learned_log | 6400 | 4.1900 | -4.1171 | 8.3072 | 4.7638 | 3.5433 |

Findings:

- All runs create $a>b$ by making $a$ positive and $b$ negative.
- Both hidden dimensions contribute positively to $\Delta$ in the analyzed runs.
- Longer constant training increases $\Delta$ strongly, but this only delays fixed-margin failure because constant multiplier still has no asymptotic correction.
- The log run needs a smaller raw $\Delta$ because $\alpha=\log n$ amplifies the margin with length.
- Learned-log runs increase $\Delta$ with more optimization, but infinite-length success depends on the product $c\Delta$, not $\Delta$ alone.

## Main Takeaway

At the weight level, the simplest model learns a very interpretable mechanism:

**The final non-target query $q_u$ aligns with the vector $k_t-k_u$, so the target key receives a higher raw score than the non-target key.**

This produces the score pattern:

```math
S_n=(a,b,b,\ldots,b),
\qquad
a>b.
```

Once this score pattern exists, the length behavior is controlled by how the attention multiplier scales the margin $\Delta$ against the growing softmax denominator.
