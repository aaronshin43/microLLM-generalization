# Stage 4B Counting Target Occurrences

## Objective

Stage 4B changes the reduced Stage 3/4A task from target presence or target identity
classification to **counting the total number of target-token occurrences**.

The model no longer answers:

```text
Is a target present?
Which target type is present?
```

It instead answers:

```text
How many target tokens are present?
```

The central question is:

**Can the strict reduced softmax-attention model length-generalize on exact counting, or does
softmax normalization make absolute multiplicity unstable?**

Stage 4A showed that the Stage 3 length-generalization story transfers to non-binary identity
classification. Counting is harder because the output must distinguish examples that all
contain targets:

```text
u u u u u    -> 0
t u u u u    -> 1
t t u u u    -> 2
t t t u u    -> 3
```

This is not only a richer classifier head. It asks whether a normalized attention readout can
represent an extensive statistic: the number of target occurrences.

## Experimental Setup

The base experiment uses the controlled single-target-type setting:

```text
target_token_count     = 1
non_target_token_count = 1
max_target_count       = 3
target_position_mode   = fixed_start
train_lengths          = [10]
test_examples          = 720
eval_chunk_examples    = 36
eval_sampling_mode     = stratified
eval_batch_size        = 8
```

Token ids:

```text
target token id    : 0
non-target token id: 1
count classes      : 0, 1, 2, 3
```

For true count $k$, the first $k$ non-final positions are targets and all remaining positions
are non-targets:

```text
k = 0: u, u, u, u, ...
k = 1: t, u, u, u, ...
k = 2: t, t, u, u, ...
k = 3: t, t, t, u, ...
```

Evaluation is stratified so each count class has the same number of examples. At
`test_examples = 720`, each count class contributes 180 examples at every evaluation length.
Long evaluation is chunked, so length 10M is evaluated without materializing the full test
tensor at once.

Output root:

```text
runs/stage4b/
```

The final reported runs use the longer e500 training budget. Earlier shorter runs are left in
the run directory as diagnostics, but they were undertrained for this task: they reached
nominal train accuracy in some cases while retaining much higher cross-entropy loss. The e500
runs reduce the train loss to about `0.05` and are the representative runs used below.

| Run | Seed | Multiplier mode | Max train steps |
|---|---:|---|---:|
| `constant_e500_t1_nt1_k3` | 42 | `constant` | 16000 |
| `log_e500_t1_nt1_k3` | 42 | `log` | 16000 |
| `learned_log_e500_t1_nt1_k3` | 42 | `learned_log` | 16000 |

The same e500 suite was also run with seeds `43` and `44` under `runs/stage4b/seed43/` and
`runs/stage4b/seed44/`.

## Model

The Stage 4B baseline intentionally keeps the Stage 4A reduced architecture and changes only
the output target.

The scoring side is unchanged:

- token embeddings are one-hot token ids,
- the final token query attends over all token keys,
- the score multiplier $\alpha$ is one of `constant`, `log`, or `learned_log`,
- attention is softmax-normalized.

The value side now produces a count-class input. For general $H$ target token types:

```text
value_output = [mass_0, ..., mass_{H-1}, nontarget_mass]
classifier   = Linear(H + 1, K + 1)
loss         = cross-entropy over count classes
```

In the base setting $H = 1$, so the classifier receives only:

```text
[target_mass, non_target_mass]
```

This is the strict baseline. It does not add a hand-coded length feature, count feature,
denominator, unnormalized attention numerator, or sum-pooling detector.

## Theory

For the single-target-type, single-non-target-type setup with true count $k$, suppose every
target has score $a$, every non-target has score $b$, and

```math
\Delta = a - b.
```

Then the total target attention mass is approximately:

```math
m_k(n) =
\frac{k e^{\alpha\Delta}}{k e^{\alpha\Delta} + (n-k)}.
```

At the training length, different counts can produce different target masses, so the model can
fit the short-length task by learning thresholds on $m_k(10)$. But $m_k(n)$ is a normalized
relative mass, not an absolute count. Its asymptotic behavior is governed jointly by length,
margin, and the multiplier mode.

With constant $\alpha = 1$:

```math
m_k(n) =
\frac{k e^\Delta}{k e^\Delta + (n-k)}
\to 0
\qquad \text{for every fixed } k.
```

All positive counts eventually become indistinguishable from count 0.

With fixed-log scaling $\alpha = \log n$:

```math
m_k(n) =
\frac{k n^\Delta}{k n^\Delta + (n-k)}.
```

The asymptotic regimes are:

```text
Delta < 1: positives collapse toward 0 target mass
Delta = 1: target mass remains count-dependent, approximately k / (k + 1)
Delta > 1: all positive counts saturate toward target mass 1
```

With learned-log scaling $\alpha = 1 + c\log(1+n)$, the same regimes are controlled by
$c\Delta$:

```text
c * Delta < 1: positives collapse toward 0 target mass
c * Delta = 1: target mass remains count-dependent
c * Delta > 1: positive counts saturate toward target mass 1
```

This is the key difference from Stage 3/4A. For presence or identity classification,
$c\Delta > 1$ is the desired asymptotic regime because it preserves target presence. For exact
counting, making the target mass merely large is not enough. The count signal is most stable
near a calibrated critical regime, not simply above a one-sided threshold.

## Training Fit

All e500 runs fit the training length perfectly and reduce the loss well below the earlier
shorter runs.

| Run | Updates | Train acc | Val acc | Train loss | Val loss |
|---|---:|---:|---:|---:|---:|
| `constant_e500_t1_nt1_k3` | 16000 | 1.000 | 1.000 | 0.0497 | 0.0494 |
| `log_e500_t1_nt1_k3` | 16000 | 1.000 | 1.000 | 0.0504 | 0.0501 |
| `learned_log_e500_t1_nt1_k3` | 16000 | 1.000 | 1.000 | 0.0497 | 0.0494 |

The target masses at the training length are cleanly separated:

| Run | Count 0 mass | Count 1 mass | Count 2 mass | Count 3 mass |
|---|---:|---:|---:|---:|
| `constant_e500_t1_nt1_k3` | 0.000 | 0.354 | 0.552 | 0.679 |
| `log_e500_t1_nt1_k3` | 0.000 | 0.353 | 0.551 | 0.678 |
| `learned_log_e500_t1_nt1_k3` | 0.000 | 0.354 | 0.552 | 0.678 |

So the model is not failing because it cannot represent the short-length training task. It
learns the count classes from the normalized mass values available at length 10.

## Results At Length 10M

At length 10M, every e500 run collapses to predicting count 0 for every example.

| Run | Accuracy | MAE | Mean predicted count | Mean target mass | Worst $\Delta$ | Learned $c$ | Worst $c\Delta$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `constant_e500_t1_nt1_k3` | 0.250 | 1.500 | 0.000 | 7.39e-7 | 1.594 | n/a | n/a |
| `log_e500_t1_nt1_k3` | 0.250 | 1.500 | 0.000 | 0.0102 | 0.691 | n/a | n/a |
| `learned_log_e500_t1_nt1_k3` | 0.250 | 1.500 | 0.000 | 9.39e-7 | 1.552 | 0.0113 | 0.0175 |

Because the test set is balanced over four count classes, always predicting 0 gives exactly
`0.250` accuracy. The confusion matrix confirms this directly:

| Run | True count 0 -> pred 0 | True count 1 -> pred 0 | True count 2 -> pred 0 | True count 3 -> pred 0 |
|---|---:|---:|---:|---:|
| `constant_e500_t1_nt1_k3` | 180 / 180 | 180 / 180 | 180 / 180 | 180 / 180 |
| `log_e500_t1_nt1_k3` | 180 / 180 | 180 / 180 | 180 / 180 | 180 / 180 |
| `learned_log_e500_t1_nt1_k3` | 180 / 180 | 180 / 180 | 180 / 180 | 180 / 180 |

The count-level target masses show what happened:

| Run | Count 1 mass @ 10M | Count 2 mass @ 10M | Count 3 mass @ 10M |
|---|---:|---:|---:|
| `constant_e500_t1_nt1_k3` | 4.93e-7 | 9.85e-7 | 1.48e-6 |
| `log_e500_t1_nt1_k3` | 0.0069 | 0.0136 | 0.0203 |
| `learned_log_e500_t1_nt1_k3` | 6.26e-7 | 1.25e-6 | 1.88e-6 |

The positive classes still have slightly different masses, especially in the fixed-log run,
but the masses have moved far outside the range seen at training length. The classifier
therefore maps all examples to the count 0 region.

## Length Sweep

The collapse appears immediately once the evaluation length leaves the training length for
constant and learned-log. Fixed-log degrades more gradually, but it also reaches the count-0
collapse regime by length 10000.

| Length | Constant acc | Constant target mass | Log acc | Log target mass | Learned-log acc | Learned-log target mass |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1.000 | 0.3960 | 1.000 | 0.3956 | 1.000 | 0.3960 |
| 100 | 0.250 | 0.0677 | 0.500 | 0.2384 | 0.250 | 0.0701 |
| 1000 | 0.250 | 0.0073 | 0.250 | 0.1403 | 0.250 | 0.0079 |
| 10000 | 0.250 | 0.0007 | 0.250 | 0.0771 | 0.250 | 0.0008 |
| 100000 | 0.250 | 7.39e-5 | 0.250 | 0.0403 | 0.250 | 8.66e-5 |
| 1000000 | 0.250 | 7.39e-6 | 0.250 | 0.0204 | 0.250 | 9.02e-6 |
| 5000000 | 0.250 | 1.48e-6 | 0.250 | 0.0126 | 0.250 | 1.85e-6 |
| 10000000 | 0.250 | 7.39e-7 | 0.250 | 0.0102 | 0.250 | 9.39e-7 |

The fixed-log run has $\Delta = 0.691 < 1$, so the theory predicts decay toward zero target
mass. That is exactly what the sweep shows. It does not fail because fixed-log attention is
inherently useless; it fails because the learned margin is in the $m_k(n) \to 0$ regime.

However, counting is more delicate than presence detection even if the margin were larger.
For exact counting, $\Delta > 1$ would push every fixed positive count toward target mass 1,
which can preserve presence but can also erase count differences among positive classes. This
means fixed-log counting would require a calibrated margin near the critical regime, not just
a comfortably large margin.

## Learned-Log Behavior

The learned-log run also fails, but for a different immediate reason: it does not learn a
meaningful log-length coefficient.

At e500:

```math
c = 0.0113,\qquad
\Delta = 1.552,\qquad
c\Delta = 0.0175.
```

This is far below the critical regime. The model therefore behaves almost like constant
attention. At the training length, the raw target masses are sufficient for perfect count
classification, so the optimizer has little direct pressure to increase $c$. This is unlike
Stage 4A, where increasing target attention helped preserve the presence/identity decision at
long length once $c\Delta$ crossed 1.

For Stage 4B, even crossing $c\Delta > 1$ would not automatically solve exact counting. It
would preserve positive mass, but it may also saturate counts 1, 2, and 3 toward the same
target-mass value. The correct condition is not simply "make $c\Delta$ large"; it is a
calibration problem.

## Multi-Seed Check

The e500 suite was repeated with seeds `42`, `43`, and `44`. All three seeds show the same
pattern: short-length fit succeeds, but length 10M collapses to count 0.

| Seed | Run | Train loss | Accuracy @ 10M | MAE @ 10M | Mean target mass @ 10M | Worst $\Delta$ | Worst $c\Delta$ |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | `constant_e500_t1_nt1_k3` | 0.0497 | 0.250 | 1.500 | 7.39e-7 | 1.594 | n/a |
| 42 | `log_e500_t1_nt1_k3` | 0.0504 | 0.250 | 1.500 | 0.0102 | 0.691 | n/a |
| 42 | `learned_log_e500_t1_nt1_k3` | 0.0497 | 0.250 | 1.500 | 9.39e-7 | 1.552 | 0.0175 |
| 43 | `constant_e500_t1_nt1_k3` | 0.0499 | 0.250 | 1.500 | 7.35e-7 | 1.589 | n/a |
| 43 | `log_e500_t1_nt1_k3` | 0.0512 | 0.250 | 1.500 | 0.0101 | 0.690 | n/a |
| 43 | `learned_log_e500_t1_nt1_k3` | 0.0499 | 0.250 | 1.500 | 9.67e-7 | 1.541 | 0.0200 |
| 44 | `constant_e500_t1_nt1_k3` | 0.0517 | 0.250 | 1.500 | 7.51e-7 | 1.610 | n/a |
| 44 | `log_e500_t1_nt1_k3` | 0.0528 | 0.250 | 1.500 | 0.0122 | 0.703 | n/a |
| 44 | `learned_log_e500_t1_nt1_k3` | 0.0518 | 0.250 | 1.500 | 9.69e-7 | 1.567 | 0.0185 |

The failure is therefore not a seed-42 accident. The exact margins and learned coefficients
move slightly, but the qualitative behavior is unchanged.

## Interpretation

Stage 4B breaks the Stage 3/4A success story in a specific way.

For presence and identity, softmax attention only needed to keep enough target mass away from
the non-target slot. Once the target mass was high enough, the task was solved. That is why
Stage 3 and Stage 4A could use the one-sided condition $c\Delta > 1$.

Counting requires more. The model must preserve the difference between one, two, and three
target occurrences. But the strict baseline does not expose an unnormalized count. It exposes
only a convex-average statistic:

```math
o(n) = [m_k(n), 1 - m_k(n)].
```

The count is present only indirectly through the normalized mass $m_k(n)$. This mass changes
with length even when the true count is fixed. As length grows, the same count moves through
the classifier's training-length decision regions and eventually collapses toward the count-0
region in the observed runs.

The important nuance is that softmax normalization does not immediately erase count
information. At length 10, the masses for counts 1, 2, and 3 are well separated, and the model
fits perfectly. The problem is that this count representation is not length-invariant. It is a
relative mass tied to $n$, $\Delta$, and $\alpha$, not a stable absolute multiplicity signal.

## Ablation 1: Unnormalized Attention Sum

The first follow-up ablation replaced the normalized softmax-mass readout with an
unnormalized numerator-sum readout:

```math
u_h =
\sum_{j:\,x_j=t_h} e^{\alpha s_j},
\qquad
u_{\text{non}} =
\sum_{j:\,x_j\ \text{non-target}} e^{\alpha s_j}.
```

The classifier still receives two values in the base $H = 1$ setting:

```text
[target numerator sum, non-target numerator sum]
```

The ablation outputs are under:

```text
runs/stage4b/ablation1_unnormalized/
```

All three e500 ablation runs fit the training length essentially perfectly:

| Run | Train acc | Val acc | Train loss | Accuracy @ 10M |
|---|---:|---:|---:|---:|
| `constant_e500_t1_nt1_k3` | 1.000 | 1.000 | 4.09e-8 | 0.250 |
| `log_e500_t1_nt1_k3` | 1.000 | 1.000 | 3.65e-8 | 0.250 |
| `learned_log_e500_t1_nt1_k3` | 1.000 | 1.000 | 4.04e-8 | 0.250 |

The training fit is stronger than the strict softmax baseline, but the long-length result is
unchanged: at 10M, every run predicts count 0 for every example.

The useful result is mechanistic. In the constant run, the **target numerator** preserves the
count factor exactly across length:

| True count | Target readout @ length 10 | Target readout @ 10M |
|---:|---:|---:|
| 1 | 57.90 | 57.90 |
| 2 | 115.80 | 115.80 |
| 3 | 173.71 | 173.71 |

So removing the softmax denominator does make target-count information available. However,
the same unnormalized readout also exposes a non-target numerator that grows with the number
of non-target tokens. In the constant run, the non-target readout grows from about `138` at
length 10 to about `1.38e8` at length 10M. The learned classifier uses this non-target feature
strongly; for example, the constant-run class-0 head has a large positive weight on the
non-target readout, while the class-3 head has a negative weight on it:

| Class | Target-readout weight | Non-target-readout weight |
|---:|---:|---:|
| 0 | -0.749 | 0.711 |
| 1 | -0.135 | 0.579 |
| 2 | 0.374 | 0.207 |
| 3 | 0.809 | -0.409 |

At long length, the non-target numerator dominates the logits and pushes every example into
class 0. The same qualitative behavior appears in fixed-log and learned-log. Their target
numerators grow, but the non-target numerator remains much larger:

| Run | Count 3 target/non-target readout ratio @ 10M |
|---|---:|
| `constant_e500_t1_nt1_k3` | 1.26e-6 |
| `log_e500_t1_nt1_k3` | 0.0190 |
| `learned_log_e500_t1_nt1_k3` | 9.30e-6 |

Numerical overflow was not the cause of failure. The recorded `readout_finite_fraction` was
`1.000` throughout the length sweep.

This ablation refines the interpretation:

**Softmax normalization is not the only bottleneck.** The target numerator does preserve count
information, but including the unnormalized non-target numerator introduces a length-growing
background scale that the classifier learns at the training length and cannot extrapolate
through. The failure shifts from "normalized target mass loses count scale" to "unnormalized
background scale overwhelms the count signal."

## Ablation 2: Target Numerator Only

The second follow-up ablation removed the length-growing non-target numerator from the
classifier input and exposed only the target numerator:

```math
u_t =
\sum_{j:\,x_j=t} e^{\alpha s_j}.
```

In the base $H = 1$ setting, the classifier receives a single scalar:

```text
[target numerator sum]
```

This is a diagnostic upper-bound style ablation. It is not a natural replacement for the
strict reduced-attention readout, because the classifier is no longer given any non-target
readout. The purpose is to isolate whether the target numerator itself contains a stable count
signal once the length-growing non-target background is removed.

The ablation outputs are under:

```text
runs/stage4b/ablation2_target_only/
```

All three e500 runs fit the training length:

| Run | Train acc | Val acc | Train loss | Accuracy @ 10M | MAE @ 10M |
|---|---:|---:|---:|---:|---:|
| `constant_e500_t1_nt1_k3` | 1.000 | 1.000 | 0.00515 | 1.000 | 0.000 |
| `log_e500_t1_nt1_k3` | 1.000 | 1.000 | 0.00539 | 0.500 | 0.750 |
| `learned_log_e500_t1_nt1_k3` | 1.000 | 1.000 | 0.00518 | 0.500 | 0.500 |

The constant run succeeds completely at 10M:

| Run | True count 0 | True count 1 | True count 2 | True count 3 |
|---|---|---|---|---|
| `constant_e500_t1_nt1_k3` | pred 0 | pred 1 | pred 2 | pred 3 |
| `log_e500_t1_nt1_k3` | pred 0 | pred 3 | pred 3 | pred 3 |
| `learned_log_e500_t1_nt1_k3` | pred 0 | pred 2 | pred 3 | pred 3 |

The reason is that, under constant scaling, the target-only readout is length-invariant. For a
true count $k$,

```math
u_t(k,n) = k e^{s_t}.
```

Since $e^{s_t}$ is fixed across lengths, the readout remains a stable linear count signal. At
10M, the constant run preserves the same count spacing:

| True count | Mean target readout @ 10M |
|---:|---:|
| 0 | 0.00 |
| 1 | 17.23 |
| 2 | 34.46 |
| 3 | 51.68 |

This is enough for the one-dimensional classifier to keep the same decision regions learned at
the training length. The normalized target attention mass still dilutes toward zero, but that
mass is no longer the classifier input in this ablation.

The fixed-log and learned-log runs fail for the opposite reason. Once the readout is
unnormalized, the length-aware multiplier changes the absolute target numerator scale:

```math
u_t(k,n) = k e^{\alpha(n) s_t}.
```

For fixed-log scaling, $\alpha(n)=\log n$, so if the learned target score $s_t$ is positive,

```math
u_t(k,n) = k n^{s_t}.
```

The target readout therefore grows with length. At 10M, this pushes smaller positive counts
past the classifier's training-length thresholds:

| Run | Count 1 target readout @ 10M | Count 2 target readout @ 10M | Count 3 target readout @ 10M |
|---|---:|---:|---:|
| `constant_e500_t1_nt1_k3` | 17.23 | 34.46 | 51.68 |
| `log_e500_t1_nt1_k3` | 1.21e9 | 2.42e9 | 3.63e9 |
| `learned_log_e500_t1_nt1_k3` | 39.50 | 79.00 | 118.50 |

Thus Ablation 2 separates two failure modes. Ablation 1 showed that the target numerator
preserves count information, but the non-target numerator creates a length-growing background
feature. Ablation 2 removes that background and confirms that the constant target numerator is
indeed a stable count signal. However, it also shows that unnormalized target numerators are
sensitive to absolute score-scale drift under length-aware multipliers. Fixed-log and
learned-log no longer collapse to count 0; instead, they overestimate positive counts because
the target numerator grows beyond the range seen during training.

The recorded `readout_finite_fraction` was `1.000` for all three runs, so this is not a
numerical overflow artifact. It is a representation issue: target-only unnormalized count
readout works when its per-target scale is length-invariant, but it is not automatically stable
under length-dependent score scaling.

## Ablation 3: Top-K Restricted Softmax

The third follow-up ablation tested a more attention-like alternative: select a fixed-size
top-k subset by corrected score, then compute a softmax only over that subset. For the base
setting, the main runs used:

```text
readout_mode = topk_softmax_mass
top_k        = 3
```

The corrected score is the raw query-key score after applying the same length-aware multiplier
used by the rest of the Stage 4B model:

```math
r_j = \alpha(n) s_j,
```

where $s_j$ is the raw score for position $j$ and $\alpha(n)$ is the configured multiplier
from `constant`, `log`, or `learned_log`. The top-k subset is selected from these corrected
scores:

```math
J_R(x) =
\operatorname{TopR}\{r_0,r_1,\dots,r_{n-1}\}.
```

This keeps the ablation aligned with the actual attention logits used by the model. The
restricted softmax then changes only the denominator set; it does not introduce a separate
ranking score or a second scoring rule.

The theoretical motivation is to keep the softmax readout but remove the length-growing
denominator. Let $R$ denote the fixed top-k size. If an example has true count $k \le R$, and
if all $k$ target positions rank above the non-target positions needed to fill the selected
set, then the top-k restricted target mass is:

```math
m_k^{\text{top-}R}
=
\frac{k e^{\alpha\Delta}}{k e^{\alpha\Delta} + (R-k)}.
```

This is the same form as the full-softmax mass, except the denominator contains $(R-k)$
instead of $(n-k)$. Since $R$ is fixed, the count signal should no longer dilute simply because
the evaluation sequence is longer. In the ideal case with `top_k = 3`, counts 0, 1, 2, and 3
would be represented by a bounded subset-level attention statistic rather than by a mass
normalized over all $n$ positions.

This theory depends on a ranking condition. The method can only preserve count if the true
target positions enter the selected subset. Therefore top-k restricted softmax has two
separate possible failure modes:

```text
ranking failure:
    target positions do not enter the top-k subset

readout failure:
    target positions enter the subset, but the restricted masses do not separate count classes
```

The intended benefit is therefore conditional: if the ranking is correct, the denominator no
longer grows with sequence length; if the ranking is wrong, the attention readout may never see
the target positions at all.

The primary seed-42 runs and follow-up seed-43/44 checks all failed:

| Seeds | Modes | Accuracy @ train length | Accuracy @ 10M | Mean top-k target count | Mean top-k non-target count |
|---|---|---:|---:|---:|---:|
| 42, 43, 44 | `constant`, `log`, `learned_log` | 0.250 | 0.250 | 0.0 | 3.0 |

The exact constant prediction differed by seed. Seed 42 predicted count 0 for every example,
while seeds 43 and 44 predicted count 2 for every example. This difference is not the main
failure mode. In every run, the selected top-k subset contained only non-target tokens, so the
classifier saw the same readout for every input:

```text
[top-k target mass, top-k non-target mass] = [0, 1]
```

This is a ranking failure, not a readout-calibration failure. The learned target-vs-non-target
margin $\Delta$ was negative in all top-k runs, so target positions ranked below non-target positions
and never entered the selected subset. The hard top-k operation creates a cold-start problem.
Once target tokens are outside the selected subset, the target readout is exactly zero, and the
model receives little or no useful gradient signal to raise target scores into the top-k set.

This ablation therefore does not show that a top-k denominator is impossible in principle. It
shows that hard top-k selection is not trainable from this cold start in the current reduced
model. A fairer test would likely need a differentiable warm-up path, such as full-softmax
pretraining followed by top-k fine-tuning, a ranking auxiliary loss, or a soft top-k relaxation
before switching to hard top-k.

## Current Conclusion

**The strict Stage 4B normalized-attention baseline fails exact count length generalization.**

The model can fit the training length, but it does not extrapolate from length 10 to long
lengths. At 10M, all final e500 runs predict count 0 for every example, giving the balanced
floor accuracy `0.250`.

The failure mode is positive-to-zero collapse:

- `constant` fails by direct attention dilution.
- `log` learns a margin below the $\Delta = 1$ critical point, so target mass still decays
  toward zero.
- `learned_log` does not learn a meaningful log coefficient under single-length training, so
  it behaves almost like constant scaling.

More importantly, Stage 4B suggests that exact counting is not solved by simply making the
Stage 3/4A target-presence mechanism stronger. Counting needs either a calibrated critical
regime or a representation that preserves count-like information more directly without also
introducing an uncontrolled length-growing background scale. The top-k ablation also shows
that bounding the denominator is not enough if the model cannot first learn the correct
target-over-non-target ranking.

## Limitations And Next Steps

This report covers only the strict softmax-normalized baseline:

- `target_token_count = 1`
- `non_target_token_count = 1`
- `max_target_count = 3`
- `target_position_mode = fixed_start`
- single training length `10`

Longer training reduces training loss but does not change the long-length failure. Training on
longer lengths or multiple lengths may improve finite-length behavior, but it does not remove
the underlying asymptotic tension of representing count through normalized mass.

The follow-up ablations now separate several mechanisms:

- the full unnormalized readout shows that target numerator scale can preserve count, but the
  non-target numerator creates a length-growing background feature,
- the target-numerator-only readout confirms that the constant target numerator is a stable
  count signal when isolated from that background,
- the hard top-k readout fails by cold-start ranking failure because target tokens never enter
  the selected subset.

Minimal candidates are:

- add denominator or log-denominator information to explicitly normalize the length-growing
  background,
- warm-start top-k from a full-softmax checkpoint or add an explicit ranking loss,
- compare against a parallel detector followed by sum pooling.

Those variants should be treated as diagnostic ablations, not as replacements for the main
Stage 4B conclusion. The current strict baseline result is that normalized softmax attention
is sufficient for presence and identity in the reduced model, but not for robust exact
counting.
