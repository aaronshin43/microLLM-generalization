# Task: Simplified Length-Aware Attention Model

## Objective

Analyze the professor's simplified length-aware attention model and connect it to our Stage 1 and Stage 2B results.

This task is not about training a new model. The goal is to write a clean theoretical note, add a small visualization, and use the result as groundwork for a future technical report.

## Simplified Setup

Use a vocabulary with two tokens:

- $t$: target token
- $u$: non-target token

For sequence length $n$, assume the input is:

```text
t, u, u, ..., u
```

Use two-dimensional embeddings:

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

Assume the final classifier uses only the last token output. For the last query row, assume the attention score vector is:

```math
S_n = (a, b, b, \ldots, b),
\qquad
a > b.
```

The target key has fixed score margin:

```math
\Delta = a - b > 0.
```

Apply an inverse temperature schedule before softmax:

```math
\operatorname{softmax}(\alpha S_n).
```

After multiplying by $X_n$, the last-token attention output is:

```math
(p_t(n), 1 - p_t(n)).
```

The target attention mass is:

```math
p_t(n)
=
\frac{e^{\alpha a}}
{e^{\alpha a} + (n-1)e^{\alpha b}}
=
\frac{e^{\alpha\Delta}}
{e^{\alpha\Delta} + (n-1)}.
```

Important correction:

The output cannot converge to $(1, 1)$. It is a convex combination of $[1, 0]$ and $[0, 1]$, so the two coordinates always sum to 1.

## Required Analysis

### 1. Constant Temperature

For:

```math
\alpha = \alpha_0,
```

derive:

```math
p_t(n)
=
\frac{e^{\alpha_0 \Delta}}
{e^{\alpha_0 \Delta} + (n-1)}
\to 0
\qquad
\text{as } n \to \infty.
```

Interpretation:

A fixed target-vs-non-target score advantage is eventually diluted by the growing number of non-target keys.

### 2. Log Temperature

For:

```math
\alpha = \log n,
```

derive:

```math
p_t(n)
=
\frac{n^\Delta}
{n^\Delta + n - 1}.
```

Then show:

```math
\Delta > 1 \Rightarrow p_t(n) \to 1,
\qquad
\Delta = 1 \Rightarrow p_t(n) \to \frac{1}{2},
\qquad
0 < \Delta < 1 \Rightarrow p_t(n) \to 0.
```

Interpretation:

A $\log n$ correction is sufficient only when the score margin is large enough.

### 3. Scaled Log Temperature

For:

```math
\alpha = c \log n,
```

derive:

```math
p_t(n)
=
\frac{n^{c\Delta}}
{n^{c\Delta} + n - 1}.
```

Then show:

```math
c\Delta > 1 \Rightarrow p_t(n) \to 1,
\qquad
c\Delta = 1 \Rightarrow p_t(n) \to \frac{1}{2},
\qquad
c\Delta < 1 \Rightarrow p_t(n) \to 0.
```

Interpretation:

The coefficient matters. A learned log correction can still fail if the effective product $c\Delta$ is too small.

### 4. General Condition

Explain how the condition is derived from:

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

So target attention converges to 1 exactly when:

```math
R_n \to 0.
```

Taking logs:

```math
\log R_n
=
\log(n-1) - \alpha\Delta.
```

Since $\log(n-1)$ behaves like $\log n$ for large $n$, $R_n \to 0$ when:

```math
\alpha\Delta - \log n \to +\infty.
```

Similarly, target attention converges to 0 when:

```math
\alpha\Delta - \log n \to -\infty.
```

Also explain the boundary case. If:

```math
\alpha\Delta - \log n \to C,
```

for some finite constant $C$, then:

```math
p_t(n)
\to
\frac{1}{1+e^{-C}}.
```

Interpretation:

The scaled target margin must beat the logarithmic growth of the number of competing non-target keys. The condition comes from asking whether the non-target-to-target ratio inside the denominator goes to 0.

## Connection To Existing Results

### Stage 1

Connect the simplified model to the Stage 1 failure:

- Stage 1 learned a fixed target-vs-non-target attention margin.
- As length increases, the softmax denominator grows with the number of non-target tokens.
- Target attention mass decreases even when the target score remains larger than each individual non-target score.
- This explains why exactly-one positive examples are eventually classified as negative.

### Stage 2B: Global Log-Temperature

Connect the simplified model to the global log-temperature intervention:

```math
\tilde{s}_{ij} = \alpha s_{ij}.
```

Here, $s_{ij}$ is the raw attention score from query position $i$ to key position $j$; in the simplified last-query case, that row is $S_n = (a,b,b,\ldots,b)$.

The simplified model predicts that this can work only if the learned effective margin satisfies a condition like:

```math
c\Delta > 1
```

for a schedule comparable to $c\log n$.

Expected interpretation:

- Global temperature is not automatically enough.
- If the learned scale is too weak, the model still suffers attention dilution.
- This matches our observed global log-temperature failure.

### Stage 2B: Target-Key Log-Bias

Connect the simplified model to the target-key bias intervention:

```math
\tilde{s}_{ij} = s_{ij} + \beta(n)r_j.
```

Here, $s_{ij}$ is the raw attention score, and $r_j$ is a learned target-likeness score for key position $j$.

The effective target-vs-non-target margin becomes approximately:

```math
\Delta_{\mathrm{eff}}(n)
=
\Delta_{\mathrm{base}}
+
\beta(n)(r_t - r_u).
```

Expected interpretation:

- Target-key bias helps because it increases the target-specific margin instead of globally scaling all scores.
- Multi-length training may help learn a better-calibrated target detector.
- This explains why `target_key_log_bias_multilength` worked better than global log-temperature.

### Finite Success Is Not Infinite Success

The simplified model should also explain why success through length 10000 is not a proof of infinite-length generalization:

- `target_attention_mean` can still decrease with length.
- positive logits can still decrease with length.
- attention entropy can still increase with length.
- the effective margin condition may fail at longer lengths.

## Deliverables

### 1. Theoretical Note

Create:

```text
infinite_generalization/documents/SIMPLIFIED_LENGTH_AWARE_ATTENTION_MODEL.md
```

The note should include:

- the simplified two-token setup
- the derivation of $p_t(n)$
- constant-temperature behavior
- $\log n$ temperature behavior
- $c\log n$ temperature behavior
- the general condition $\alpha\Delta - \log n \to +\infty$
- the convex-combination correction
- the connection to Stage 1 and Stage 2B

Use the explanatory style of:

```text
infinite_generalization/documents/THEORETICAL_MODEL.md
```

### 2. Visualization Script Or Notebook

Add a small self-contained visualization.

Preferred script:

```text
infinite_generalization/src/plot_simplified_length_attention.py
```

Optional notebook:

```text
infinite_generalization/notebooks/simplified_length_aware_attention_model.ipynb
```

The visualization should plot $p_t(n)$ over increasing $n$ for:

- constant $\alpha$
- $\alpha = \log n$ with $\Delta < 1$
- $\alpha = \log n$ with $\Delta = 1$
- $\alpha = \log n$ with $\Delta > 1$
- $\alpha = c\log n$ where $c\Delta < 1$
- $\alpha = c\log n$ where $c\Delta > 1$

Recommended output:

```text
infinite_generalization/documents/figures/simplified_length_aware_attention_pt.png
```

Use a log-scaled x-axis.

### 3. Stage 2B Analysis Update

Update:

```text
infinite_generalization/documents/STAGE2B_LENGTH_AWARE_ATTENTION_ANALYSIS.md
```

Add a concise section linking the empirical Stage 2B results to the simplified model.

The section should state:

- global log-temperature failed because the learned effective margin was too weak
- target-key bias worked better because it increased target-specific margin
- multi-length training likely helped calibrate the target-key detector
- finite-length success does not imply an infinite-length guarantee

### 4. Optional Formula Tests

If the formula is implemented as reusable code, add tests for:

- $p_t(n)$ is always between 0 and 1
- $(p_t(n), 1 - p_t(n))$ sums to 1
- constant $\alpha$ decays toward 0
- $c\Delta > 1$ moves toward 1
- $c\Delta < 1$ moves toward 0
- $c\Delta = 1$ approaches 1/2

If the formula exists only inside a plotting script, tests are optional.

## Success Criteria

This task is complete when:

- the theoretical note is written
- the limiting cases are mathematically correct
- the convex-combination correction is clearly stated
- at least one figure illustrates the regimes
- the Stage 1 and Stage 2B connections are explicit
- the document can be reused in a future technical report

## Execution Order

1. Write `SIMPLIFIED_LENGTH_AWARE_ATTENTION_MODEL.md`.
2. Add the visualization script or notebook.
3. Save the figure under `documents/figures/`.
4. Update `STAGE2B_LENGTH_AWARE_ATTENTION_ANALYSIS.md`.
5. Add formula tests only if reusable formula code is introduced.
