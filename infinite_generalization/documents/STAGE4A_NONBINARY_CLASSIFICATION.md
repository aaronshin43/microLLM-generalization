# Stage 4A Non-Binary Classification

## Objective

Stage 4A changes the reduced Stage 3 task from a **binary existential detector** into a **non-binary identity classifier**.

In every previous Stage 3 experiment, the model only had to decide whether a target was present:

```text
positive: sequence contains a target token
negative: sequence contains no target token
```

Stage 4A keeps the same length-aware reduced attention model but replaces the output. Instead of present-vs-absent, the model must report **which target token type is present**, or a dedicated $n$ ("none") class when no target appears:

```text
class 0 .. H-1 : the sequence contains target type t_0 .. t_{H-1}
class n (= H)  : the sequence contains no target token
```

The central question is:

**Does the Stage 3 length-generalization story (constant fails, fixed-log succeeds, learned-log succeeds once $c\Delta > 1$) still hold when the head must identify the target type, not just detect presence — and does the multi-class structure introduce a new failure mode?**

## Why Stage 4A

Stage 3E already added multiple target token types, but it kept the readout binary: the value pathway only carried $(p_{\text{target}}, 1 - p_{\text{target}})$, so the head still answered a single present-vs-absent question. Identity was never read out.

Stage 4A removes that simplification. The value pathway now carries per-target-type attention mass, and the head is a genuine $(H + 1)$-class classifier. This is the first Stage where the model is asked to *name* the target rather than to *detect* it, which is the more realistic existential-classification task.

## Experimental Setup

This first Stage 4A experiment is intentionally controlled and mirrors the Stage 3E base setup:

```text
target_token_count     = 3
non_target_token_count = 1
target_position_mode   = fixed_start
train_lengths          = [10]
test_examples          = 720
eval_chunk_examples    = 36
eval_sampling_mode     = stratified
eval_batch_size        = 8
```

Token ids:

```text
target token ids : 0, 1, 2
non-target id    : 3
none class index : 3
```

Positive examples place one target type at position 0; negative examples contain only the non-target token:

```text
positive (type h): t_h, u, u, ..., u
negative         : u,   u, u, ..., u
```

Each evaluation length is generated in chunks of `eval_chunk_examples` to avoid materializing the full tensor at 10M, and positive examples are stratified so each target token id is generated evenly (no random-draw bias). This is the same stratified protocol adopted across the Stage 3 reruns.

### Value Pathway And Head

It helps to line this up directly against the "Stage 3 Implementation" section of `STAGE3_SIMPLIFIED_LENGTH_AWARE_ATTENTION.md`. Everything in the **scoring half** of the model is identical there and here. The query/key projections, the last-query-only read, the score multiplier $\alpha$, and the attention weights

```math
A_j = \frac{e^{\alpha s_j}}{\sum_k e^{\alpha s_k}}
```

are unchanged, so the dilution formula $m_h(n) = e^{\alpha\Delta_h}/(e^{\alpha\Delta_h} + (n-1))$ still governs how much attention mass lands on the target position. Stage 4A only changes the **value half**: the shape of the attention output $o(n)$ and the head that reads it.

**Change 1 — the attention output gains one slot per target type.**

In Stage 3 the token values are fixed 2-dim one-hot vectors, so the attention output $o(n) = \sum_j A_j x_j$ has exactly two slots, target mass and non-target mass:

```math
t \mapsto [1,0],\quad u \mapsto [0,1]
\quad\Longrightarrow\quad
o(n) = \big(\underbrace{p_t}_{\text{target}},\ \underbrace{1-p_t}_{\text{non-target}}\big).
```

Stage 4A has $H$ target types, so the fixed value becomes an $(H+1)$-dim one-hot: one basis vector per target type, plus one shared slot for every non-target token. The same $o(n) = \sum_j A_j x_j$ now reads out as

```math
t_h \mapsto e_h,\quad u \mapsto e_H
\quad\Longrightarrow\quad
o(n) = \big(\underbrace{m_0, m_1, \ldots, m_{H-1}}_{\text{per-type target mass}},\ \underbrace{m_{\text{non}}}_{\text{non-target mass}}\big),
```

where each slot is the attention mass landing on tokens of that type:

```math
m_h = \!\!\sum_{j:\,x_j = t_h}\!\! A_j,
\qquad
m_{\text{non}} = \!\!\sum_{j:\,x_j\ \text{non-target}}\!\! A_j.
```

This is precisely the Stage 3 readout with the single "target mass" slot **split into per-type masses**. With $H = 1$ it reduces to $(m_0, m_{\text{non}}) = (p_t, 1 - p_t)$, recovering the Stage 3 binary output identically.

**Change 2 — the head becomes a multi-class classifier.**

Because the output dimension changed, the head changes with it:

| | Stage 3 (binary) | Stage 4A (non-binary) |
|---|---|---|
| value dim | 2 | $H + 1$ |
| head | $z = w^\top o + b_{\text{cls}}$ → one scalar | `Linear(H+1, H+1)` → $H+1$ class logits |
| decision | $z \ge 0$ → positive | $\arg\max$ over $H+1$ classes |
| loss | binary cross-entropy | cross-entropy |
| label | present / absent | positive → type $h$; negative → none class $H$ |

Stage 3 decided present-vs-absent from the sign of a single logit $z(n)$. Stage 4A produces one logit per class and takes the $\arg\max$, so it answers *which* target type is present, or $n$ (none) when no target appears.

**Worked example ($H = 3$).**

The value dimension is 4, so $o(n) = [m_0, m_1, m_2, m_{\text{non}}]$. For a positive example whose target is type $t_1$:

```math
o(n) \approx [\,0,\ p_t,\ 0,\ 1-p_t\,].
```

The slots $m_0$ and $m_2$ are **structurally zero**, because tokens $t_0$ and $t_2$ never appear in this sequence — only $t_1$ and non-targets do. The classifier maps these four values to four logits, and the correct class is $1$.

When dilution sets in (for example, constant mode at long length, $p_t \to 0$), the output drifts to

```math
o(n) \to [\,0,\ 0,\ 0,\ 1\,]
\quad\Longrightarrow\quad
\text{predicted class } n.
```

Because $m_0$ and $m_2$ can never grow, a failing positive can only collapse onto the $n$ slot — never onto a wrong target type. This is the structural origin of the two axes used throughout the rest of this document: the **presence axis** ($\sum_h m_h$ versus $m_{\text{non}}$) inherits the Stage 3 dilution and is the fragile one, while the **identity axis** (which $m_h$ is largest) rides on top of it and never fails on its own. The empirical confirmation is in "Graceful Failure: Presence Collapse, Not Type Confusion" below.

## Theory

For a positive example of target type $h$, the final (non-target) query produces the score row

```math
S_n^{(h)} = (a_h, b, b, \ldots, b),
\qquad
\Delta_h = a_h - b,
```

and the attention mass on the single target position is the familiar dilution expression

```math
m_h(n) = \frac{e^{\alpha\Delta_h}}{e^{\alpha\Delta_h} + (n-1)}.
```

The corresponding non-target slot for this controlled single-non-target setup is

```math
m_{\text{non}}(n)
= 1 - m_h(n)
= \frac{n-1}{e^{\alpha\Delta_h} + (n-1)}.
```

Equivalently, the non-target-to-target ratio is

```math
\frac{m_{\text{non}}(n)}{m_h(n)} = (n-1)e^{-\alpha\Delta_h}.
```

Stage 4A inherits this mechanism unchanged. The **presence axis** introduced in the previous section is exactly this dilution term, now written per target type as $m_h(n)$. The **identity axis** sits on top of it: in the controlled setup each positive contains exactly one target type, so the correct $m_h$ is the only non-zero target slot as long as the target position keeps any non-trivial mass.

The key prediction follows directly:

**The identity axis cannot fail before the presence axis fails.** Once $m_h$ dilutes below $m_{\text{non}}$, the example collapses to the $n$ class — it is not reassigned to a *wrong* target type. Failure should appear as positive→none collapse, not as type confusion.

The multiplier modes and their asymptotic diagnostics are unchanged:

```math
\text{constant: } \alpha = 1,
\qquad
\text{fixed-log: } \alpha = \log n,
\qquad
\text{learned-log: } \alpha = 1 + c\log(1 + n),
```

with the learned-log success condition

```math
c\,\Delta_{\min} > 1,
\qquad
\Delta_{\min} = \min_h \Delta_h.
```

## Base Experiment

Output root:

```text
runs/stage4a/
```

| Run | Multiplier mode | Max train steps |
|---|---|---:|
| `constant_e100_t3_nt1` | `constant` | 3200 |
| `log_e50_t3_nt1` | `log` | 1600 |
| `learned_log_e200_t3_nt1` | `learned_log` | 6400 |
| `learned_log_e300_t3_nt1` | `learned_log` | 9600 |

All four runs reach train and validation accuracy `1.000` at the training length, so every difference below is a pure length-extrapolation effect, not a fitting difference.

| Run | Updates | Train acc | Val acc | Train loss |
|---|---:|---:|---:|---:|
| `constant_e100_t3_nt1` | 3200 | 1.000 | 1.000 | 0.0065 |
| `log_e50_t3_nt1` | 1600 | 1.000 | 1.000 | 0.0287 |
| `learned_log_e200_t3_nt1` | 6400 | 1.000 | 1.000 | 0.0010 |
| `learned_log_e300_t3_nt1` | 9600 | 1.000 | 1.000 | 0.0002 |

## Base Results At Length 10M

| Run | Accuracy | None-class acc | Positive-correct frac | Target attention | Mean $\Delta_{\min}$ | Worst $\Delta_{\min}$ | Worst $c\Delta_{\min}$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `constant_e100_t3_nt1` | 0.500 | 1.000 | 0.000 | 0.0009 | 9.070 | 8.922 | n/a |
| `log_e50_t3_nt1` | 1.000 | 1.000 | 1.000 | 1.0000 | 4.262 | 4.110 | n/a |
| `learned_log_e200_t3_nt1` | 1.000 | 1.000 | 1.000 | 0.9982 | 8.456 | 8.301 | 0.860 |
| `learned_log_e300_t3_nt1` | 1.000 | 1.000 | 1.000 | 0.9999 | 8.670 | 8.518 | 1.018 |

The qualitative pattern matches the rest of Stage 3:

- Constant multiplier fits short lengths but fails at long length.
- Fixed-log multiplier succeeds because every target-type margin is well above 1.
- Learned-log succeeds across the tested range, but whether it does so *for the right reason* depends on training, as analyzed below.

## Constant Multiplier Behavior

The constant run is perfect up to length 1000 and then dilutes:

| Length | Accuracy | Positive-correct frac | Predicted-none frac (positives) | Target attention |
|---:|---:|---:|---:|---:|
| 10 | 1.000 | 1.000 | 0.000 | 0.9989 |
| 1000 | 1.000 | 1.000 | 0.000 | 0.8955 |
| 10000 | 0.667 | 0.333 | 0.667 | 0.4653 |
| 100000 | 0.500 | 0.000 | 1.000 | 0.0813 |
| 10000000 | 0.500 | 0.000 | 1.000 | 0.0009 |

With $\alpha = 1$ fixed, target attention and non-target mass are

```math
m_h(n) = \frac{e^{\Delta_h}}{e^{\Delta_h} + (n-1)},
\qquad
m_{\text{non}}(n) = 1 - m_h(n).
```

As $n$ grows, $m_{\text{non}}(n) \to 1$ and $m_h(n) \to 0$, so the target slot collapses. Accuracy floors at `0.500`: negatives are still classified as $n$ perfectly (none-class acc stays `1.000`), but every positive collapses to $n$, so exactly half the balanced test set is wrong.

## Graceful Failure: Presence Collapse, Not Type Confusion

The multi-class decomposition confirms the central Stage 4A prediction. As the constant run fails, positives are reassigned to the $n$ class, **never** to a wrong target type:

| Length | Predicted-none frac (positives) | Predicted-other-target frac (positives) |
|---:|---:|---:|
| 10000 | 0.667 | 0.000 |
| 100000 | 1.000 | 0.000 |
| 10000000 | 1.000 | 0.000 |

The identity axis is robust; the bottleneck is purely the presence axis. Moreover, the collapse is **margin-ordered** across target types — the type with the largest margin survives longest:

| Length | $t_0$ recall ($\Delta = 9.356$) | $t_1$ recall ($\Delta = 8.922$) | $t_2$ recall ($\Delta = 8.932$) |
|---:|---:|---:|---:|
| 1000 | 1.000 | 1.000 | 1.000 |
| 10000 | 1.000 | 0.000 | 0.000 |
| 100000 | 0.000 | 0.000 | 0.000 |

At length 10000 only $t_0$ — the largest-margin type — still survives, which is exactly why the aggregate positive-correct fraction is `0.333` there. This is the multi-class analogue of the Stage 3D/3E worst-margin diagnostic: **the class is only as robust as its smallest-margin target type, and that type fails first.**

## Fixed-Log Behavior

The fixed-log run succeeds at every evaluated length. Because $\alpha = \log n$,

```math
m_h(n) = \frac{n^{\Delta_h}}{n^{\Delta_h} + (n-1)},
\qquad
m_{\text{non}}(n) = 1 - m_h(n).
```

With $\Delta_{\text{worst}} \approx 4.11 \gg 1$, the non-target mass vanishes as roughly $n^{1-\Delta_{\text{worst}}} \approx n^{-3.1}$. Target attention stays `1.000` and accuracy stays `1.000` through 10M. This is the only mode whose success is guaranteed by a comfortable margin over the threshold.

## Learned-Log Behavior And The $c\Delta$ Crossing

The learned-log result is best understood through the threshold $c\Delta_{\min} > 1$.

**e200 (6400 steps).** This run is accurate at every evaluated length, including 10M, but it does **not** reach the asymptotic success condition:

```math
c = 0.1036,\quad \Delta_{\min} = 8.301,\quad c\,\Delta_{\min} = 0.860 < 1.
```

Because $c\Delta_{\min} < 1$, target attention is still slowly decreasing with length (`0.9998 -> 0.9982` from length 10 to 10M). The useful diagnostic is the non-target-to-target ratio:

```math
r_{\text{non}/h}(n) \approx e^{-\Delta_{\min}} n^{1-c\Delta_{\min}}
= e^{-8.301} n^{0.140}.
```

The exponent is positive, so the ratio eventually grows. The prefactor is tiny, so the expected failure point is pushed far beyond the benchmark, around length $10^{25}$. Thus e200 passes the 10M evaluation, but it should still fail asymptotically.

**e300 (9600 steps).** The rerun increases both the learned coefficient and the worst raw margin enough to cross the threshold:

```math
c = 0.1195,\quad \Delta_{\min} = 8.518,\quad c\,\Delta_{\min} = 1.018 > 1.
```

Now the exponent $1-c\Delta_{\min}$ is negative, so the non-target ratio decays rather than grows. Target attention no longer drifts downward and remains flat at `0.9999` at 10M. This is the successful learned-log case.

| Run | $c$ | $\Delta_{\min}$ (worst) | $c\Delta_{\min}$ | Target attention @ 10M |
|---|---:|---:|---:|---:|
| `learned_log_e200_t3_nt1` | 0.1036 | 8.301 | 0.860 | 0.9982 (drifting down) |
| `learned_log_e300_t3_nt1` | 0.1195 | 8.518 | 1.018 | 0.9999 (flat) |

This is the intended contrast: e200 is empirically correct through 10M but below the asymptotic threshold, while e300 crosses $c\Delta_{\min} > 1$ and becomes the genuine learned-log success case.

## Interpretation

Stage 4A does not change the underlying length-generalization mechanism; the softmax dilution term

```math
m_h(n) = \frac{e^{\alpha\Delta_h}}{e^{\alpha\Delta_h} + (n-1)}
```

is identical to Stage 3. What Stage 4A adds is a clean separation of the two things the model must do:

- **Presence** (target vs. $n$) is the fragile axis. It is governed by dilution and is where every failure occurs.
- **Identity** (which target type) is the robust axis. It never fails on its own; positives collapse to $n$ rather than to a wrong type.

Because identity rides entirely on top of presence, the non-binary head inherits the Stage 3 conclusions without weakening them: constant fails, fixed-log succeeds, and learned-log succeeds once $c\Delta_{\min} > 1$. The smallest-margin target type remains the bottleneck and is the first to fall.

## Current Conclusion

**Stage 4A succeeds in the controlled base setup, and the Stage 3 length-generalization story transfers to the non-binary classifier.**

- Constant scaling fails at long length via positive→none collapse.
- Fixed-log succeeds with a comfortable margin.
- Learned-log e200 passes the $10^7$ benchmark but with $c\Delta_{\min} = 0.860 < 1$, so it is asymptotically incomplete (target attention is still slowly diluting).
- Learned-log e300 crosses the threshold with $c\Delta_{\min} = 1.018 > 1$ and reaches a genuinely flat, length-invariant attention profile.

The main new finding is structural:

**In the multi-class setting the failure mode is presence collapse, not type confusion, and the smallest-margin target type fails first.**

## Limitations

- Only `target_token_count = 3` and `non_target_token_count = 1` were tested.
- The target was fixed at position 0 (`fixed_start`).
- Only one seed was analyzed.
- Positive examples contained exactly one target token; multiple target occurrences and mixed-type sequences were not tested.
- The $c\Delta_{\min} \approx 1.018$ crossing in e300 is only modestly above threshold; it was not stress-tested with more target types or non-target types.
- This result applies only to the reduced, no-positional-encoding model and does not automatically transfer to the full Stage 1/2 transformer.
