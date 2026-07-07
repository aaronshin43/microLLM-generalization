# When Does Length-Aware Attention Generalize in a Reduced Binary Classifier?

## Abstract

Softmax attention classifiers trained on short sequences can fail to generalize
to much longer sequences, even on simple target-token detection tasks. This
report studies that failure mode in a deliberately reduced binary attention
classifier. The model sees sequences containing either one target token or only
non-target tokens, and a single final query attends over the sequence before a
binary classifier makes the prediction. We compare constant, fixed-log, and
learned-log score multipliers to test when length-aware attention scaling
overcomes the growing softmax denominator. Let $\Delta$ denote the score margin
between the target and non-target tokens, and let $c$ denote the learned
coefficient in the learned-log multiplier. The trained model satisfies the
two-score assumption: one target score and one shared non-target score.
Therefore, the closed-form target-attention equation accurately describes the
learned model. The experiments show three regimes. A constant multiplier
improves with more training but fails asymptotically. A fixed-log multiplier
succeeds when $\Delta>1$. A learned-log multiplier succeeds only after
optimization pushes the effective product $c\Delta$ above 1. Thus, passing a
finite long-sequence evaluation is not sufficient evidence of infinite-length
generalization; the effective margin must grow faster than the softmax
denominator.

## Introduction

Length generalization is a basic difficulty for sequence models. A classifier can
fit short training sequences while relying on a mechanism that does not remain
valid at longer sequence lengths. This problem appears even in simple
existential target-token tasks, where the model only needs to decide whether a
sequence contains a particular token.

The motivating failure mode is attention dilution. If one target token competes
against many non-target tokens in a softmax denominator, a fixed target advantage
over each individual non-target token may still become too weak as the number of
non-target tokens grows. This suggests a simple theoretical question: can
length-aware attention scaling make the target advantage grow quickly enough to
overcome the length-growing denominator?

This report focuses on a reduced binary classifier proposed for that question.
The model is intentionally much simpler than a full transformer. It uses fixed
one-hot token values, learned query and key projections, a single final-query
attention readout, and a binary classifier. The goal is not to show that this
model is a competitive architecture. The goal is to test whether the simplified
closed-form attention theory correctly describes a trainable model in the
controlled setting where its assumptions can be directly checked.

The main result is that the reduced model does satisfy the required two-score
structure after training. On positive examples, the final-query attention scores
take the form

```math
S_n=(a,b,b,\ldots,b),
```

where $a$ is the target score, $b$ is the shared non-target score, and
$\Delta=a-b$. Once this structure holds, the target attention mass is determined
by

```math
p_t(n)=\frac{e^{\alpha\Delta}}{e^{\alpha\Delta}+(n-1)}.
```

The experiments then behave as the theory predicts. Constant attention scaling
fails at long length. Fixed log-length scaling succeeds when the learned margin
is larger than 1. Learned log-length scaling succeeds asymptotically only after
training makes $c\Delta>1$. This last point is important: a model can pass a
finite evaluation length such as 10 million while still being predicted to fail
at much larger lengths if $c\Delta<1$.

## Background: Simplified Binary Attention

The task uses a two-token vocabulary:

- $t$: target token
- $u$: non-target token

A positive length-$n$ sequence has one target token followed by non-target
tokens:

```text
t, u, u, ..., u
```

A negative length-$n$ sequence contains only non-target tokens:

```text
u, u, u, ..., u
```

The values are fixed one-hot vectors:

```math
t \mapsto [1,0],
\qquad
u \mapsto [0,1].
```

The model reads from the final query. Since the final token is $u$, the final
query is the non-target query. Under the two-score assumption, the final-query
score row for a positive example is

```math
S_n=(a,b,b,\ldots,b),
```

where $a$ is the score assigned to the target key and $b$ is the shared score
assigned to every non-target key. Define the margin

```math
\Delta=a-b.
```

Before the softmax, the model applies a score multiplier $\alpha$. The target
attention mass is

```math
p_t(n)
=
\frac{e^{\alpha a}}
{e^{\alpha a}+(n-1)e^{\alpha b}}.
```

Dividing by $e^{\alpha b}$ gives the closed form

```math
p_t(n)
=
\frac{e^{\alpha\Delta}}
{e^{\alpha\Delta}+(n-1)}
=
\frac{1}{1+(n-1)e^{-\alpha\Delta}}.
```

The attention output is then

```math
o(n)=(p_t(n),1-p_t(n)).
```

Thus the output is a convex combination of target evidence and non-target
evidence.

### Length-Scaling Regimes

The denominator term $(n-1)$ is the source of length dependence. If
$\alpha=1$, then

```math
p_t(n)
=
\frac{e^\Delta}{e^\Delta+(n-1)}
\to 0
\qquad
\text{as } n\to\infty.
```

A fixed margin can beat each non-target token individually, but it cannot beat
an unbounded number of non-target competitors.

If $\alpha=\log n$, then

```math
p_t(n)
=
\frac{n^\Delta}{n^\Delta+n-1}.
```

Therefore

```math
\Delta>1 \Rightarrow p_t(n)\to1,
\qquad
\Delta=1 \Rightarrow p_t(n)\to\frac12,
\qquad
0<\Delta<1 \Rightarrow p_t(n)\to0.
```

For the learned-log multiplier used in the experiments,

```math
\alpha=1+c\log(1+n),
```

the asymptotic condition is approximately

```math
c\Delta>1.
```

A single general condition summarizes these cases. Target attention converges to
1 when

```math
\alpha\Delta-\log n\to+\infty.
```

The scaled target margin must grow faster than the logarithm of the number of
competing non-target keys.

## Experimental Design

The experiment implements the simplified model as a trainable reduced attention
classifier. It uses:

- fixed one-hot token values $t\mapsto[1,0]$ and $u\mapsto[0,1]$
- learned query projection $W_Q$
- learned key projection $W_K$
- a final-token query only
- softmax attention over all sequence positions
- a binary classifier on the attention output

For input matrix $X$, the model computes

```math
Q=XW_Q,
\qquad
K=XW_K.
```

The final-query score for position $j$ is

```math
s_j
=
\frac{q_{\mathrm{last}}^\top k_j}{\sqrt d}.
```

The multiplier $\alpha$ is then applied before softmax. The experiments compare
three multiplier modes:

| Mode | Multiplier |
|---|---|
| `constant` | $\alpha=1$ |
| `log` | $\alpha=\log n$ |
| `learned_log` | $\alpha=1+c\log(1+n)$ |

For `learned_log`, the coefficient is parameterized as

```math
c=\mathrm{softplus}(k_\alpha),
```

so it remains positive during optimization.

All analyzed runs train at length 10 and evaluate up to length 10,000,000. Long
evaluation is chunked to avoid materializing the full evaluation tensor at once.
The analyzed runs are:

```text
runs/stage3base/constant_e50
runs/stage3base/constant_e100
runs/stage3base/constant_e1000
runs/stage3base/log_e50
runs/stage3base/learned_log_e50
runs/stage3base/learned_log_e100
runs/stage3base/learned_log_e200
```

The suffix `e50`, for example, denotes a 50-epoch training run.

## Assumption Validation

The closed-form expression for $p_t(n)$ is exact only if the two-score
assumption holds. Therefore, the first empirical question is not whether the
formula can be algebraically derived. The first empirical question is whether
the trained model actually produces one target score and one shared non-target
score.

The answer is yes in all analyzed Stage 3 runs. The non-target score standard
deviation is 0.0 at every evaluated length. The reconstructed closed-form
attention mass also matches the empirical attention mass up to small numerical
error.

| Run | Non-target score std | Max observed attention error | Two-score assumption |
|---|---:|---:|---|
| `constant_e50` | 0.0 | about $2.5\times10^{-8}$ | holds |
| `constant_e100` | 0.0 | about $2.9\times10^{-8}$ | holds |
| `constant_e1000` | 0.0 | about $9.0\times10^{-7}$ | holds |
| `log_e50` | 0.0 | about $1.2\times10^{-7}$ | holds |
| `learned_log_e50` | 0.0 | about $1.4\times10^{-3}$ | holds |
| `learned_log_e100` | 0.0 | about $1.5\times10^{-5}$ | holds |
| `learned_log_e200` | 0.0 | about $1.2\times10^{-7}$ | holds |

This validation step is central to the report. It shows that the theory is not
being applied after the fact to an unrelated black-box model. The trained model
learns the exact score structure required by the simplified analysis.

## Results

Negative examples are classified correctly in every analyzed run and length.
This is expected: a negative sequence contains only non-target values, so the
attention output remains in the non-target direction regardless of how attention
is distributed over positions. The length-generalization failure is therefore a
positive-example failure: the target mass in positive examples can dilute until
the classifier no longer detects the target.

Table 1 summarizes the main results at evaluation length 10,000,000.

| Run | Updates | $\Delta$ | Learned $c$ | $c\Delta$ | $p_t(10M)$ | Positive logit at 10M | Positive accuracy at 10M |
|---|---:|---:|---:|---:|---:|---:|---:|
| `constant_e50` | 1600 | 8.1006 | n/a | n/a | 0.0003 | -3.6580 | 0.0000 |
| `constant_e100` | 3200 | 8.9883 | n/a | n/a | 0.0008 | -4.9271 | 0.0000 |
| `constant_e1000` | 32000 | 12.2702 | n/a | n/a | 0.0209 | -18.6624 | 0.0000 |
| `log_e50` | 1600 | 3.9361 | n/a | n/a | 1.0000 | 3.7196 | 1.0000 |
| `learned_log_e50` | 1600 | 7.4085 | 0.0661 | 0.4894 | 0.3039 | -1.3771 | 0.0000 |
| `learned_log_e100` | 3200 | 7.9306 | 0.0961 | 0.7623 | 0.9837 | 4.8629 | 1.0000 |
| `learned_log_e200` | 6400 | 8.2994 | 0.1352 | 1.1218 | 1.0000 | 6.8194 | 1.0000 |

![Target attention by length](../infinite_generalization/documents/figures/stage3_target_attention_by_length.png)

**Figure 1:** Target attention as a function of evaluation length. Constant
scaling eventually dilutes target mass. Fixed log scaling succeeds because the
learned margin is above 1. Learned-log behavior depends on whether optimization
pushes $c\Delta$ above 1.

![Positive logit by length](../infinite_generalization/documents/figures/stage3_positive_logit_by_length.png)

**Figure 2:** Positive logit as a function of evaluation length. Positive
accuracy fails when the target attention mass moves past the classifier's
decision boundary.

### Constant Scaling

Constant scaling uses $\alpha=1$. The observed pattern is that more training
increases $\Delta$, which moves the failure point outward, but does not change
the asymptotic regime. The target attention mass remains

```math
p_t(n)=\frac{e^\Delta}{e^\Delta+(n-1)}.
```

For any fixed $\Delta$, $p_t(n)\to0$ as $n\to\infty$. Thus constant scaling can
learn a stronger finite-length solution but not an infinite-length solution.

This distinction is visible in the e1000 run. It learns a much larger margin
than e50 or e100, and therefore keeps more target attention at length 10M. But
the positive logit is still negative at 10M, and the theory predicts eventual
failure for any finite fixed margin.

### Fixed Log Scaling

The fixed-log run uses $\alpha=\log n$. The learned margin is
$\Delta\approx3.9361$, comfortably above the threshold $\Delta>1$. Therefore the
theory predicts $p_t(n)\to1$, which is exactly what is observed. Target
attention reaches 1.0000 at long lengths, the positive logit remains positive,
and positive accuracy remains 1.0000 through length 10M.

### Learned Log Scaling

The learned-log runs show that finite evaluation success is not the same as
asymptotic success. The 50-epoch run has $c\Delta=0.4894$ and fails at 10M. The
100-epoch run has $c\Delta=0.7623$. It succeeds at 10M, but since
$c\Delta<1$, the theory predicts eventual failure at sufficiently larger
lengths. The 200-epoch run reaches $c\Delta=1.1218>1$, entering the asymptotic
success regime predicted by the simplified model.

This is one of the main lessons of the experiment. A finite benchmark can be
too short to distinguish a strong finite-length solution from an infinite-length
solution. The better diagnostic in this reduced setting is the effective
exponent $c\Delta$.

## Mechanism: What The Model Learns

The reduced model also allows a direct weight-level explanation of where
$\Delta$ comes from. Since the final input token is the non-target token $u$, the
final query is $q_u$. Let $k_t$ be the target key and $k_u$ be the non-target
key. Then

```math
a
=
\frac{q_u^\top k_t}{\sqrt d},
\qquad
b
=
\frac{q_u^\top k_u}{\sqrt d}.
```

The margin is therefore

```math
\Delta
=
a-b
=
\frac{q_u^\top(k_t-k_u)}{\sqrt d}.
```

For the `learned_log_e200` checkpoint, the learned vectors are approximately

```math
q_u=
\begin{bmatrix}
2.1707\\
1.8260
\end{bmatrix},
\qquad
k_t=
\begin{bmatrix}
1.5457\\
1.4076
\end{bmatrix},
\qquad
k_u=
\begin{bmatrix}
-1.5580\\
-1.3366
\end{bmatrix}.
```

This gives

```math
a\approx4.1900,
\qquad
b\approx-4.1171,
\qquad
\Delta\approx8.3072.
```

Geometrically, $q_u$ points in a direction that separates the target key from
the non-target key. The target key has a positive projection along the final
query direction, while the non-target key has a negative projection. Equivalently,
$q_u$ aligns with the difference vector $k_t-k_u$. This alignment creates the
score pattern $a>b$ required by the two-score theory.

This mechanism explains how the model creates a target advantage, but it also
shows why a target advantage is not by itself enough. Under constant scaling,
even a large fixed $\Delta$ is eventually overwhelmed by the growing number of
non-target positions. Learned-log attention has an additional degree of freedom:
it can increase the coefficient $c$ so that the effective margin grows like
$c\Delta\log n$. In the successful learned-log run, optimization makes the
product $c\Delta$ cross the threshold.

## Discussion

The main contribution of this report is a controlled verification of the
simplified length-aware attention theory in a trainable reduced model. The
closed-form target attention equation is not merely a post-hoc curve fit. The
trained model satisfies the two-score assumption directly, so the theory applies
to the learned weights.

The result also clarifies the difference between finite and asymptotic
generalization. In constant mode, longer training increases the raw margin and
pushes the failure length farther away, but it never changes the limit behavior.
In learned-log mode, a run can pass length 10M while still having $c\Delta<1$.
This makes $c\Delta$ a more informative diagnostic than accuracy at a single
large evaluation length.

The reduced model is intentionally limited. It uses fixed one-hot values, no
positional encodings, one final query, and a simple binary classifier. These
restrictions make the mechanism easy to analyze, but they also mean that the
conclusion should not be transferred directly to full transformers. A full
transformer may have non-identical non-target scores, learned value vectors,
multiple heads, residual streams, feed-forward layers, layer normalization, and
pooling effects. Those components can break the exact two-score structure that
makes the present analysis clean.

The value of the reduced model is therefore diagnostic. It separates two
questions that are entangled in a full transformer. First, can the architecture
create a target-vs-non-target margin? Second, does the length-scaling mechanism
make that margin grow fast enough to beat the softmax denominator? In this
controlled binary setting, the answer to both questions can be measured
directly.

## Conclusion

This report studied length generalization in a reduced binary attention
classifier. The trained model learns the two-score structure assumed by the
simplified theory, making the closed-form target attention expression an
accurate description of the learned model.

The experiments support three conclusions. Constant attention scaling learns
finite-length robustness but fails asymptotically. Fixed log-length scaling
succeeds when the target-vs-non-target margin satisfies $\Delta>1$. Learned-log
scaling succeeds asymptotically only when optimization pushes $c\Delta>1$.
Therefore, in this reduced setting, the central condition for length
generalization is not merely fitting the training length or passing a finite
long-length benchmark. The effective target margin must grow faster than the
logarithm of the number of competing non-target keys.

## Appendix A: Additional Material To Add Later

The final version may include some of the following supporting material if space
allows:

- full $W_Q$ and $W_K$ matrices for the mechanism example
- a two-dimensional query-key vector plot for $q_u$, $k_t$, $k_u$, and
  $k_t-k_u$
- classifier threshold estimates for the constant runs
- the multi-length training negative result as a short appendix
- a one-paragraph note that later Stage 4A and Stage 4B experiments extended the
  reduced framework beyond binary classification, while the main report focuses
  on the binary setting for clarity
