# When Does Length-Aware Attention Generalize in a Reduced Binary Classifier?

## Abstract

_jmac Don't use so much detailed technical notation and definitions in the abstract. Try to describe the conclusion and findings in prose rather than symbols and formulas. jmac_ 
Softmax attention classifiers trained on short sequences can fail to generalize to much longer sequences, even on simple target-token detection tasks. This report studies that failure mode in a deliberately reduced binary attention classifier. The model sees sequences containing either one target token or only non-target tokens, and a single final query attends over the sequence before a binary classifier makes the prediction. We compare constant, fixed-log, and learned-log score multipliers to test when length-aware attention scaling overcomes the growing softmax denominator. Let $\Delta$ denote the score margin between the target and non-target tokens, and let $c$ denote the learned coefficient in the learned-log multiplier. The trained model satisfies the two-score assumption: one target score and one shared non-target score. Therefore, the closed-form target-attention equation accurately describes the learned model. The experiments show three regimes. A constant multiplier improves with more training but fails asymptotically. A fixed-log multiplier succeeds when $\Delta>1$. A learned-log multiplier succeeds only after optimization pushes the effective product $c\Delta$ above 1. Thus, passing a finite long-sequence evaluation is not sufficient evidence of infinite-length _jmac We may want to refer to unbounded length rather than infinite length jmac_ generalization; the effective margin must grow faster than the softmax
denominator.

## Introduction

_jmac A similar comment to the abstract. This introduction introduces too much technical detail that is not explained. The reader cannot understand the issue with the softmax denominator until the model is more clear. Add some additional explanation here but leave the technical details until later.  jmac_ 
Length generalization is a basic difficulty for sequence models. A classifier can fit short training sequences while relying on a mechanism that does not remain valid at longer sequence lengths. This problem appears even in simple existential target-token tasks, where the model only needs to decide whether a sequence contains a particular token.

The motivating failure mode is attention dilution. If one target token competes against many non-target tokens in a softmax denominator, a fixed target advantage over each individual non-target token may still become too weak as the number of non-target tokens grows. This suggests a simple theoretical question: can length-aware attention scaling make the target advantage grow quickly enough to overcome the length-growing denominator?

This report focuses on a reduced binary classifier proposed for that question. The model is intentionally much simpler than a full transformer. It uses fixed one-hot token values, learned query and key projections, a single final-query attention readout, and a binary classifier. The goal is not to show that this model is a competitive architecture. The goal is to test whether the simplified closed-form attention theory correctly describes a trainable model in the controlled setting where its assumptions can be directly checked.

The main result is that the reduced model does satisfy the required two-score structure after training. On positive examples, the final-query attention scores take the form

```math
S_n=(a,b,b,\ldots,b),
```

where $a$ is the target score, $b$ is the shared non-target score, and $\Delta=a-b$. Once this structure holds, the target attention mass is determined by

```math
p_t(n)=\frac{e^{\alpha\Delta}}{e^{\alpha\Delta}+(n-1)}.
```

The experiments then behave as the theory predicts. Constant attention scaling fails at long length. Fixed log-length scaling succeeds when the learned margin
is larger than 1. Learned log-length scaling succeeds asymptotically only after training makes $c\Delta>1$. This last point is important: a model can pass a
finite evaluation length such as 10 million while still being predicted to fail at much larger lengths if $c\Delta<1$.

_jmac Please add a related work section. This can be based on a single piece of literature if desired. See the instructions on "Related Work" at https://dnulab.org/internal/report-guidelines jmac_

## Background: Simplified Binary Attention

The task uses a two-token vocabulary:

- $t$: target token
- $u$: non-target token

A positive length-$n$ sequence, with $n\geq2$, contains one target token and
$n-1$ non-target tokens. The token in the last position is always the
non-target token $u$, so positive and negative examples use the same readout
query. The model has no positional encoding, so permuting the tokens among the
first $n-1$ positions does not change its output. We place the target in the
first position only to make the notation easier to read; any position other
than the last is equivalent.

```text
t, u, u, ..., u
```

A negative length-$n$ sequence contains only non-target tokens:

```text
u, u, u, ..., u
```

At a high level, the query at the last position scores every token in the
sequence, softmax converts those scores into attention weights, and the
weighted sum of the token values is passed to a binary classifier.

The model begins with fixed one-hot representations for $t$ and $u$. Learned
query and key projections produce the attention scores. In the value pathway,
the same one-hot representations are used directly, with no learned value
projection:

```math
t \mapsto [1,0],
\qquad
u \mapsto [0,1].
```

This fixed value pathway makes the output directly record how much total
attention is assigned to each token type. For a sequence represented by the
one-hot matrix $X$, the learned projections produce $Q=XW_Q$ and $K=XW_K$.
Conventional self-attention forms the full $n\times n$ score matrix
$QK^\top/\sqrt d$. This reduced classifier uses only the query at the last
position and therefore needs only its scores against the $n$ keys:

```math
s_j=\frac{q_{\mathrm{last}}^\top k_j}{\sqrt d},
\qquad j=1,\ldots,n.
```

Here $d$ is the query and key dimension. Since the token in the last position
is $u$, $q_{\mathrm{last}}=q_u$ for every example. The target and non-target
scores are therefore

```math
a=\frac{q_u^\top k_t}{\sqrt d},
\qquad
b=\frac{q_u^\top k_u}{\sqrt d}.
```

The architecture thus fixes the score vector of a positive example to the form

```math
S_n=(a,b,b,\ldots,b),
```

while training determines the values of $a$ and $b$. Define the target score
margin as

```math
\Delta=a-b.
```

The central change in this study is an additional score multiplier
$\alpha(n)$ applied before softmax. The attention weight on position $j$ is

```math
A_j
=
\frac{e^{\alpha(n)s_j}}
{\sum_{k=1}^{n}e^{\alpha(n)s_k}}.
```

Standard scaled dot-product attention corresponds to $\alpha(n)=1$ here,
because the usual $1/\sqrt d$ factor is already included in $s_j$. The
length-aware variants instead make $\alpha(n)$ a fixed or learned function of
the sequence length. This additional multiplier is the main departure from the
usual attention computation.

For the positive score vector $S_n=(a,b,\ldots,b)$, the attention mass on the
single target key is

```math
p_t(n)
=
\frac{e^{\alpha(n)a}}
{e^{\alpha(n)a}+(n-1)e^{\alpha(n)b}}.
```

Dividing the numerator and denominator by $e^{\alpha(n)b}$ gives the closed
form

```math
p_t(n)
=
\frac{e^{\alpha(n)\Delta}}
{e^{\alpha(n)\Delta}+(n-1)}.
```

The classifier reads the weighted sum of the fixed value vectors:

```math
o(n)=\sum_{j=1}^{n} A_j v_j.
```

On a positive example, the target carries weight $p_t(n)$ and all non-targets
together carry the complementary weight $1-p_t(n)$. Its output is therefore

```math
o_{\text{pos}}(n)
=
p_t(n)\,[1,0]+\big(1-p_t(n)\big)\,[0,1]
=
\big(p_t(n),\,1-p_t(n)\big).
```

A negative example contains only $u$, so its output is always

```math
o_{\text{neg}}(n)=[0,1].
```

The positive output is thus governed by the single scalar $p_t(n)$. If
$p_t(n)\to0$, the positive representation converges to the negative
representation, and the representation margin between the two classes
vanishes. If $p_t(n)$ converges to a constant strictly between 0 and 1,
classification depends on the learned classifier's decision boundary. The
stronger, target-dominant outcome is $p_t(n)\to1$. The conditions below
distinguish these three cases.

### Length-Scaling Regimes

Assume that $\Delta>0$, so the target has a score advantage over each
non-target. If $\Delta\leq0$, none of the positive multipliers considered below
can create such an advantage. The competing term $(n-1)$ is the source of
length dependence.

For the constant baseline $\alpha(n)=1$,

```math
p_t(n)
=
\frac{e^\Delta}{e^\Delta+(n-1)}
\to 0
\qquad
\text{as } n\to\infty.
```

A fixed margin can beat each non-target token individually, but it cannot beat
an unbounded number of non-target competitors. Consequently,
$o_{\text{pos}}(n)\to[0,1]$, the same representation as
$o_{\text{neg}}(n)$.

For the fixed-log multiplier $\alpha(n)=\log n$,

```math
p_t(n)
=
\frac{n^\Delta}{n^\Delta+n-1}.
```

The limit depends on the learned margin:

```math
\lim_{n\to\infty}p_t(n)
=
\begin{cases}
1, & \Delta>1,\\
\tfrac12, & \Delta=1,\\
0, & 0<\Delta<1.
\end{cases}
```

Thus, $\Delta>1$ is the exact condition for target attention to converge to 1,
while $0<\Delta<1$ causes the positive representation to collapse onto the
negative representation. At the boundary $\Delta=1$, the two representations
remain distinct, and classification depends on the learned decision boundary.

For the learned-log multiplier used in the experiments,

```math
\alpha(n)=1+c\log(1+n),
```

the target attention mass is

```math
p_t(n)
=
\frac{e^\Delta(1+n)^{c\Delta}}
{e^\Delta(1+n)^{c\Delta}+(n-1)}.
```

Its limit is

```math
\lim_{n\to\infty}p_t(n)
=
\begin{cases}
1, & c\Delta>1,\\
\dfrac{e^\Delta}{e^\Delta+1}, & c\Delta=1,\\
0, & c\Delta<1.
\end{cases}
```

Therefore, $c\Delta>1$ is the exact condition for target attention to
converge to 1. The equality case again has a nonzero limiting target mass and
depends on the classifier's decision boundary, whereas $c\Delta<1$ produces
asymptotic collapse.

A single general condition summarizes all three regimes. Rewriting the target
attention mass as

```math
p_t(n)
=
\frac{1}
{1+\exp\!\left(\log(n-1)-\alpha(n)\Delta\right)},
```

and defining

```math
g(n)=\alpha(n)\Delta-\log(n-1),
```

shows that $g(n)$ fully determines the asymptotic target mass: $p_t(n)\to1$
when $g(n)\to+\infty$, $p_t(n)\to0$ when $g(n)\to-\infty$, and
$p_t(n)\to1/(1+e^{-L})$ when $g(n)\to L$ for some finite $L$. Target-dominant
attention therefore requires
the scaled target margin to exceed the logarithm of the number of competing
non-target keys by an amount that grows without bound.

## Experimental Design

The experiment implements the simplified model as a trainable reduced attention classifier. It uses:

- one-hot token inputs
- fixed semantic value vectors $t\mapsto[1,0]$ and $u\mapsto[0,1]$
- learned query projection $W_Q$
- learned key projection $W_K$
- a last-token query only
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

The multiplier $\alpha$ is then applied before softmax. The experiments compare three multiplier modes:

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

All analyzed runs train at length 10 and evaluate up to length 10,000,000. Long evaluation is chunked to avoid materializing the full evaluation tensor at once.

_jmac Don't include the directory names here. Explain the labels you will be using on later figures. jmac_

Run labels combine the multiplier mode with the training budget. For example,
`constant_e50` denotes constant scaling trained for 50 epochs, while
`learned_log_e200` denotes learned-log scaling trained for 200 epochs. We
analyze constant runs at 50, 100, and 1000 epochs; one fixed-log run at 50
epochs; and learned-log runs at 50, 100, and 200 epochs. These compact labels
are used in Table 1 and both figures below.

## Closed-Form Consistency Check

The closed-form expression for $p_t(n)$ is exact only if the two-score assumption holds. Therefore, the first empirical question is not whether the formula can be algebraically derived. The first empirical question is whether the trained model actually produces one target score and one shared non-target score.

As expected from the two-token construction, the non-target score standard
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

Because the vocabulary has only two token identities and no positional
encoding, this agreement is expected rather than a separate empirical finding.
The check primarily verifies that the implementation, metric extraction, and
closed-form calculation are consistent before the formula is used to interpret
the longer-length results.
_jmac I think this is overstated. This is really just a debugging step. After all, we have only two tokens, so only two scores can be learned. It is not at all surprising that the error is zero except for tiny numerical deviations. jmac_

## Results

Negative examples are classified correctly in every analyzed run and length. This is expected: a negative sequence contains only non-target values, so the attention output remains in the non-target direction regardless of how attention is distributed over positions. The length-generalization failure is therefore a positive-example failure: the target mass in positive examples can dilute until the classifier no longer detects the target.

Table 1 summarizes the main results at evaluation length 10,000,000. The
`Updates` column is the total number of optimizer steps, not the number of
training examples. With 2000 examples and batch size 64, each epoch contains 32
updates; therefore 50, 100, 200, and 1000 epochs correspond to 1600, 3200,
6400, and 32000 updates, respectively.

| Run | Updates | $\Delta$ | Learned $c$ | $c\Delta$ | $p_t(10M)$ | Positive logit at 10M | Positive accuracy at 10M |
|---|---:|---:|---:|---:|---:|---:|---:|
| `constant_e50` | 1600 | 8.10 | n/a | n/a | 0.0003 | -3.66 | 0 |
| `constant_e100` | 3200 | 8.99 | n/a | n/a | 0.0008 | -4.93 | 0 |
| `constant_e1000` | 32000 | 12.27 | n/a | n/a | 0.021 | -18.66 | 0 |
| `log_e50` | 1600 | 3.94 | n/a | n/a | 1.000 | 3.72 | 1 |
| `learned_log_e50` | 1600 | 7.41 | 0.066 | 0.49 | 0.304 | -1.38 | 0 |
| `learned_log_e100` | 3200 | 7.93 | 0.096 | 0.76 | 0.984 | 4.86 | 1 |
| `learned_log_e200` | 6400 | 8.30 | 0.135 | 1.12 | 1.000 | 6.82 | 1 |

_jmac Explain what the updates column means. Use fewer decimal places for most of this. jmac_


![Target attention by length](./figures/final_report_target_attention_by_length.png)
_jmac fixed path jmac_

**Figure 1:** Target attention as a function of evaluation length. Constant scaling eventually dilutes target mass. Fixed log scaling succeeds because the
learned margin is above 1. Learned-log behavior depends on whether optimization pushes $c\Delta$ above 1.

_jmac In the figures, fix the graph titles so that they do not refer to stage 3, jmac_


![Positive logit by length](./figures/final_report_positive_logit_by_length.png)
_jmac fixed path jmac_

**Figure 2:** Positive logit as a function of evaluation length. Positive accuracy fails when the target attention mass moves past the classifier's decision boundary.

### Constant Scaling

Constant scaling uses $\alpha=1$. The observed pattern is that more training increases $\Delta$, which moves the failure point outward, but does not change the asymptotic regime. The target attention mass remains

```math
p_t(n)=\frac{e^\Delta}{e^\Delta+(n-1)}.
```

For any fixed $\Delta$, $p_t(n)\to0$ as $n\to\infty$. Thus constant scaling can learn a stronger finite-length solution but not an infinite-length solution.

This distinction is visible in the e1000 run. It learns a much larger margin than e50 or e100, and therefore keeps more target attention at length 10M. But the positive logit is still negative at 10M, and the theory predicts eventual failure for any finite fixed margin.

### Fixed Log Scaling

The fixed-log run uses $\alpha=\log n$. The learned margin is _jmac Use fewer decimal places here and elsewhere. jmac_
$\Delta\approx3.94$, comfortably above the threshold $\Delta>1$. Therefore the theory predicts $p_t(n)\to1$, which is exactly what is observed. Target attention reaches 1.000 at long lengths, the positive logit remains positive, and positive accuracy remains 1 through length 10M.

### Learned Log Scaling

The learned-log runs show that finite evaluation success is not the same as asymptotic success. The 50-epoch run has $c\Delta=0.49$ and fails at 10M. The 100-epoch run has $c\Delta=0.76$. It succeeds at 10M, but since $c\Delta<1$, the theory predicts eventual failure at sufficiently larger lengths. The 200-epoch run reaches $c\Delta=1.12>1$, entering the asymptotic success regime predicted by the simplified model.

This is one of the main lessons of the experiment. A finite benchmark can be too short to distinguish a strong finite-length solution from an infinite-length solution. The better diagnostic in this reduced setting is the effective exponent $c\Delta$.

## Mechanism: What The Model Learns

The reduced model also allows a direct weight-level explanation of where $\Delta$ comes from. Since the last input token is the non-target token $u$, the final query is $q_u$. Let $k_t$ be the target key and $k_u$ be the non-target key. Then

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
2.171\\
1.826
\end{bmatrix},
\qquad
k_t=
\begin{bmatrix}
1.546\\
1.408
\end{bmatrix},
\qquad
k_u=
\begin{bmatrix}
-1.558\\
-1.337
\end{bmatrix}.
```

This gives

```math
a\approx4.190,
\qquad
b\approx-4.117,
\qquad
\Delta\approx8.307.
```

Geometrically, $q_u$ points in a direction that separates the target key from the non-target key. The target key has a positive projection along the final query direction, while the non-target key has a negative projection. Equivalently, $q_u$ aligns with the difference vector $k_t-k_u$. This alignment creates the score pattern $a>b$ required by the two-score theory.

This mechanism explains how the model creates a target advantage, but it also shows why a target advantage is not by itself enough. Under constant scaling, even a large fixed $\Delta$ is eventually overwhelmed by the growing number of non-target positions. Learned-log attention has an additional degree of freedom: it can increase the coefficient $c$ so that the effective margin grows like $c\Delta\log n$. In the successful learned-log run, optimization makes the product $c\Delta$ cross the threshold.

## Discussion

The main contribution of this report is a controlled analysis of length-aware
attention scaling in a trainable reduced model. The two-token,
position-independent construction makes the two-score form expected by design;
it is not itself a learned discovery. This simplification lets us apply the
closed-form target-attention equation directly and isolate the effect of the
score multiplier from other transformer components.

The result also clarifies the difference between finite and asymptotic generalization. In constant mode, longer training increases the raw margin and pushes the failure length farther away, but it never changes the limit behavior. In learned-log mode, a run can pass length 10M while still having $c\Delta<1$. This makes $c\Delta$ a more informative diagnostic than accuracy at a single large evaluation length.

The reduced model is intentionally limited. It uses fixed semantic value vectors, no positional encodings, one final query, and a simple binary classifier. These restrictions make the mechanism easy to analyze, but they also mean that the conclusion should not be transferred directly to full transformers. A full transformer may have non-identical non-target scores, learned value vectors, multiple heads, residual streams, feed-forward layers, layer normalization, and pooling effects. Those components can break the exact two-score structure that makes the present analysis clean. The value of the reduced model is therefore diagnostic. It separates two questions that are entangled in a full transformer. First, can the architecture create a target-vs-non-target margin? Second, does the length-scaling mechanism
make that margin grow fast enough to beat the softmax denominator? In this controlled binary setting, the answer to both questions can be measured directly.

## Conclusion

This report studied length generalization in a reduced binary attention classifier. Its two-token, position-independent construction yields the two-score structure assumed by the simplified theory, making the closed-form target-attention expression an accurate description of the model.

The experiments support three conclusions. Constant attention scaling learns finite-length robustness but fails asymptotically. Fixed log-length scaling succeeds when the target-vs-non-target margin satisfies $\Delta>1$. Learned-log scaling succeeds asymptotically only when optimization pushes $c\Delta>1$. Therefore, in this reduced setting, the central condition for length generalization is not merely fitting the training length or passing a finite long-length benchmark. The effective target margin must grow faster than the logarithm of the number of competing non-target keys.

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
