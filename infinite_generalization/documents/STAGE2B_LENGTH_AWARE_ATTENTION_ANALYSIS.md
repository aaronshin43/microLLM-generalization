# Stage 2B Length-Aware Attention Analysis

## Objective

This document summarizes the Stage 2B experiments that tested whether learned length-aware attention corrections can prevent the Stage 1 transformer's long-length failure.

The motivating question was whether the model can learn a length-dependent correction of the form:

```math
k \log(1 + n)
```

where `n` is the sequence length and `k` is a learned scalar parameter.

The reason this correction is natural is that target attention must compete against a softmax denominator whose number of terms grows with sequence length.

Stage 2B tested two intervention families:

- `global_log_temperature`: multiply all attention logits by a learned length-dependent scale
- `target_key_log_bias`: add a learned length-dependent bias to keys that look target-like

Each family was tested under two training conditions:

- fixed length-10 training
- short multi-length calibration on `[10, 20, 50, 100]`

## Implementation Summary

Stage 2B keeps the Stage 1 transformer backbone intentionally close to the original model:

- one transformer encoder layer
- one attention head
- `d_model = 64`
- no positional encoding
- max pooling over encoded token states
- binary classifier on the pooled vector

The main implementation change is that self-attention is computed manually instead of using `nn.MultiheadAttention` directly. This exposes the attention score before softmax:

```math
s_{ij}
=
\frac{q_i \cdot k_j}{\sqrt{d_{\mathrm{head}}}}
```

Here:

- `i` is the query position
- `j` is the key position
- `q_i` is the query vector at position `i`
- `k_j` is the key vector at position `j`
- `d_head` is the attention head dimension

### Global Log-Temperature

The global temperature variant multiplies every attention score by the same length-dependent factor:

```math
\tilde{s}_{ij}
=
\alpha(n) s_{ij}
```

with:

```math
\alpha(n)
=
1
+
\mathrm{softplus}(k_{\alpha}) \log(1+n)
```

The interpretation is simple: longer sequences make the attention softmax sharper everywhere.

This is a weak intervention because it does not know which key is the target key.

### Target-Key Log-Bias

The target-key bias variant first computes a learned target-like score for each key:

```math
r_j
=
w_{\mathrm{target}}^\top k_j
+
b_{\mathrm{target}}
```

Then it adds a length-dependent bias to each key:

```math
\tilde{s}_{ij}
=
s_{ij}
+
\beta(n) r_j
```

with:

```math
\beta(n)
=
\mathrm{softplus}(k_{\beta}) \log(1+n)
```

Here `r_j` is learned from the key representation only. It does not use the true target mask or label at evaluation time.

The interpretation is also simple: longer sequences give more score advantage to keys that the model judges to be target-like.

### Corrected Attention Mass

Both variants replace the original attention score `s_ij` with a corrected score:

```math
\tilde{s}_{ij}
```

The actual attention weight is then:

```math
a_{ij}
=
\frac{
\exp(\tilde{s}_{ij})
}{
\sum_{\ell=1}^{n} \exp(\tilde{s}_{i\ell})
}
```

For a target key position `t`, the relevant quantity is:

```math
a_{it}
=
\frac{
\exp(\tilde{s}_{it})
}{
\exp(\tilde{s}_{it})
+
\sum_{\ell \ne t} \exp(\tilde{s}_{i\ell})
}
```

The purpose of Stage 2B is to make `tilde{s}_{it}` large enough that the target numerator remains competitive as the non-target denominator grows with `n`.

Summary: **The intervention does not change the softmax itself; it changes the scores that enter the softmax denominator.**

Summary: **Stage 2B changes only the attention score calculation, while keeping the rest of the small transformer classifier as close as possible to Stage 1.**

## Runs

Analyzed run directories:

```text
runs/stage2b_global_log_temperature
runs/stage2b_target_key_log_bias
runs/stage2b_global_log_temperature_multilength
runs/stage2b_target_key_log_bias_multilength
```

Long-length evaluation used `eval_batch_size = 1` for completed runs because attention memory and runtime scale quadratically with sequence length.

## High-Level Finding

The only fully successful Stage 2B run was:

```math
\text{target-key log-bias}
+
\text{multi-length calibration}
```

Summary: **A learned log-length correction can help, but it needs both target-specific biasing and length variation during training.**

## Overall Accuracy

| Run | Training Lengths | Long-Length Result |
|---|---:|---|
| `stage2b_global_log_temperature` | `[10]` | exactly-one positives fail from length 1500 |
| `stage2b_target_key_log_bias` | `[10]` | exactly-one positives fail from length 5000 |
| `stage2b_global_log_temperature_multilength` | `[10, 20, 50, 100]` | exactly-one positives fail at length 10000 |
| `stage2b_target_key_log_bias_multilength` | `[10, 20, 50, 100]` | succeeds through length 10000 |

Summary: **The target-key bias helps more than global scaling, and multi-length training is necessary for the cleanest extrapolation.**

## Length-Sweep Results

### Fixed Length-10 Global Log-Temperature

| Length | Overall Accuracy | Positive Accuracy | Negative Accuracy |
|---:|---:|---:|---:|
| 1000 | 1.0000 | 1.0000 | 1.0000 |
| 1500 | 0.5009 | 0.0018 | 1.0000 |
| 2000 | 0.5000 | 0.0000 | 1.0000 |
| 5000 | 0.5000 | 0.0000 | 1.0000 |
| 10000 | 0.5000 | 0.0000 | 1.0000 |

Summary: **Global temperature scaling preserves negatives but fails sparse positives once length becomes too large.**

### Fixed Length-10 Target-Key Log-Bias

| Length | Overall Accuracy | Positive Accuracy | Negative Accuracy |
|---:|---:|---:|---:|
| 1000 | 1.0000 | 1.0000 | 1.0000 |
| 1500 | 1.0000 | 1.0000 | 1.0000 |
| 2000 | 1.0000 | 1.0000 | 1.0000 |
| 5000 | 0.5000 | 0.0000 | 1.0000 |
| 10000 | 0.5000 | 0.0000 | 1.0000 |

Summary: **Target-specific bias extends the failure point, but fixed length-10 training still does not learn enough extrapolating margin.**

### Multi-Length Global Log-Temperature

| Length | Overall Accuracy | Positive Accuracy | Negative Accuracy |
|---:|---:|---:|---:|
| 1000 | 1.0000 | 1.0000 | 1.0000 |
| 1500 | 1.0000 | 1.0000 | 1.0000 |
| 2000 | 1.0000 | 1.0000 | 1.0000 |
| 5000 | 1.0000 | 1.0000 | 1.0000 |
| 10000 | 0.5000 | 0.0000 | 1.0000 |

Summary: **Multi-length training delays the collapse, but global sharpening alone still fails at the longest tested length.**

### Multi-Length Target-Key Log-Bias

| Length | Overall Accuracy | Positive Accuracy | Negative Accuracy |
|---:|---:|---:|---:|
| 1000 | 1.0000 | 1.0000 | 1.0000 |
| 1500 | 1.0000 | 1.0000 | 1.0000 |
| 2000 | 1.0000 | 1.0000 | 1.0000 |
| 5000 | 1.0000 | 1.0000 | 1.0000 |
| 10000 | 1.0000 | 1.0000 | 1.0000 |

Summary: **This is the first transformer run that keeps exactly-one positives and negatives correct through length 10000.**

## Diagnostic Slice Results

At length 10000:

| Run | Exactly-One Positive | Negative | k=3 Positive | k=10 Positive | Density Positive |
|---|---:|---:|---:|---:|---:|
| `global_log_temperature` | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| `target_key_log_bias` | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| `global_log_temperature_multilength` | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `target_key_log_bias_multilength` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The successful run also solved all exactly-one positional slices at length 10000:

| Slice | Accuracy |
|---|---:|
| positive single target near beginning | 1.0000 |
| positive single target near middle | 1.0000 |
| positive single target near end | 1.0000 |

Summary: **Many-target positives can hide failure, so the decisive result is that target-key bias plus multi-length training solves exactly-one positives in every target region.**

## Learned Length Scale

The learned length-scale parameter stayed small in all runs.

| Run | Learned Positive Scale | Correction at Length 10000 |
|---|---:|---:|
| `global_log_temperature` | 0.00691 | `alpha ~= 1.064` |
| `target_key_log_bias` | 0.00706 | `beta ~= 0.065` |
| `global_log_temperature_multilength` | 0.00695 | `alpha ~= 1.064` |
| `target_key_log_bias_multilength` | 0.00721 | `beta ~= 0.066` |

The global multiplier only sharpened logits by about 6 percent at length 10000.

Summary: **The model did not learn a large global `k`; the winning run succeeds because the small correction is applied to a very strong target-specific detector.**

## Target-Key Detector

For target-key bias runs, the target detector learned to assign a much larger target-like score to the true target token than to all non-target tokens.

| Run | Target Score | Max Non-Target Score | Margin |
|---|---:|---:|---:|
| `target_key_log_bias` | 6.094 | 1.085 | 5.009 |
| `target_key_log_bias_multilength` | 8.817 | 1.030 | 7.786 |

The multi-length target-key bias run learned a stronger separation between the target token and the most target-like non-target token.

Summary: **The successful model learns a clean target-key detector, not merely a generic attention sharpening rule.**

## Attention Behavior

For a length-10000 exactly-one positive example:

| Run | Logit | Probability | Target Attention Mean | Target Attention Max |
|---|---:|---:|---:|---:|
| `global_log_temperature` | -3.941 | 0.019 | 0.095 | 0.403 |
| `target_key_log_bias` | -2.898 | 0.052 | 0.166 | 0.611 |
| `global_log_temperature_multilength` | -1.629 | 0.164 | 0.158 | 0.736 |
| `target_key_log_bias_multilength` | 1.792 | 0.857 | 0.377 | 0.981 |

Summary: **The successful run keeps substantially more attention mass on the target key at length 10000, which keeps the final logit positive.**

## Attention Trend In The Successful Run

The successful `target_key_log_bias_multilength` run still shows target attention dilution as length increases.

For exactly-one positive random-target examples:

| Length | Logit | Probability | Target Attention Mean | Target Attention Max | Attention Entropy Mean | Target Corrected Rank Mean |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 10.138 | 1.000 | 0.910 | 1.000 | 0.188 | 1.100 |
| 100 | 11.774 | 1.000 | 0.945 | 1.000 | 0.365 | 1.125 |
| 500 | 10.688 | 1.000 | 0.811 | 0.999 | 1.435 | 1.120 |
| 1000 | 8.603 | 1.000 | 0.725 | 0.998 | 2.154 | 1.131 |
| 1500 | 6.759 | 0.999 | 0.657 | 0.997 | 2.741 | 1.130 |
| 2000 | 5.876 | 0.997 | 0.613 | 0.996 | 3.149 | 1.127 |
| 5000 | 3.293 | 0.964 | 0.464 | 0.990 | 4.615 | 1.064 |
| 10000 | 1.783 | 0.856 | 0.374 | 0.981 | 5.683 | 1.066 |

The target key remains near rank 1 after correction, and the maximum target attention stays very high. However, mean target attention decreases from about `0.91` at length 10 to about `0.37` at length 10000. The positive logit also decreases from about `10.14` to about `1.78`.

Summary: **The successful run is robust through 10000, but it is not evidence of true infinite-length generalization because the target attention mean and positive logit margin are still declining.**

The same trend is visible across target positions at length 10000:

| Slice | Logit | Target Attention Mean | Target Attention Max | Attention Entropy Mean |
|---|---:|---:|---:|---:|
| target near beginning | 1.766 | 0.369 | 0.981 | 5.729 |
| target near middle | 1.782 | 0.372 | 0.981 | 5.705 |
| target near end | 1.836 | 0.376 | 0.981 | 5.664 |

This shows that the run is not relying on a target-position shortcut. The behavior is similar for beginning, middle, and end target positions.

Summary: **Position is not the main issue; the remaining risk is length-driven dilution, not where the target appears.**

## Extrapolation Risk Beyond 10000

The successful run should be interpreted as a strong finite-length success, not a proof of infinite-length generalization.

The reason is:

```math
\mathrm{target\_attention\_mean}(n)
\text{ decreases as } n \text{ grows}
```

```math
\mathrm{positive\_logit}(n)
\text{ decreases as } n \text{ grows}
```

```math
\mathrm{attention\_entropy}(n)
\text{ increases as } n \text{ grows}
```

At length 10000 the model still has a positive logit margin:

```math
\mathrm{logit}(10000) \approx 1.78
```

and:

```math
\sigma(\mathrm{logit}(10000)) \approx 0.856
```

but this margin is much smaller than at shorter lengths. If the same trend continues, the model may eventually cross the zero-logit decision boundary at longer lengths.

Summary: **Stage 2B target-key bias multi-length solved the tested range, but the trend suggests it may still fail at sufficiently longer lengths.**

## Interpretation

The Stage 2B results support four conclusions.

1. A global `k log(n)` multiplier is not enough by itself.
2. Target-specific length correction is more effective than global attention sharpening.
3. Multi-length calibration is needed to learn a target detector and correction that extrapolate to much longer tested lengths.
4. The successful run still shows declining target attention and logit margin, so it should not be treated as a proof of infinite-length generalization.

The most important comparison is:

```math
\text{global log-temperature + multi-length}
\rightarrow
\text{fails at } n = 10000
```

while:

```math
\text{target-key log-bias + multi-length}
\rightarrow
\text{succeeds at } n = 10000
```

This means the problem is not only that attention needs to become sharper with length. The model also needs to know **which key** should receive the length-dependent advantage.

Summary: **The length correction must be targeted; otherwise the model can sharpen attention without reliably preserving the sparse target signal.**

## Answer To The Stage 2B Question

Can a learned `k log(n)` correction solve the long-length failure?

Answer:

```text
Yes, but not as a simple global multiplier.
```

The successful mechanism was:

```math
\text{target-key log-bias}
+
\text{multi-length calibration}
```

This suggests that future transformer-preserving interventions should focus on target-specific or detector-style attention biases rather than only global softmax temperature scaling.

Summary: **The experiment validates the length-aware attention idea, but the correction must be target-aware and trained with some length variation.**
