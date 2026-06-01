# Stage 2B Task: Length-Aware Attention Interventions

## Objective

Test whether the Stage 1 transformer's long-length failure can be mitigated by adding a learned length-dependent correction to attention.

The motivating question is:

> If target attention needs roughly a `log(n)`-scale score advantage to remain dominant as sequence length `n` grows, can the model learn a `k log(n)` correction at an appropriate place in the network?

Run two intervention families:

1. global learned `log(n)` attention temperature
2. learned target-key `log(n)` attention bias

The goal is not only to improve accuracy. The goal is to determine whether the Stage 1 failure mechanism is actually caused by finite attention-score margins and softmax denominator growth.

## Background

For a query row, attention to the target key is:

```text
target_mass = exp(score_target) / sum_j exp(score_j)
```

As sequence length grows, the denominator receives more non-target terms. If the target score margin stays finite, the target mass can decay even when `score_target` remains high.

To keep target attention mass stable under longer lengths, the target score advantage must roughly compensate for the growing number of competing keys:

```text
target_score - typical_non_target_score ~= log(n)
```

Stage 1 was trained only at length 10, so the model had little pressure to create a margin large enough for lengths such as 500, 1000, or 10000.

## Core Hypotheses

### Hypothesis A: Global Log-Temperature Can Help If The Target Is Already The Top Key

Scale all attention logits by a learned length-dependent multiplier:

```text
base_score(i, j) = q_i dot k_j / sqrt(d_head)
alpha(n) = 1 + softplus(k) * log(n)
score(i, j) = alpha(n) * base_score(i, j)
```

Expected behavior:

- if the target key usually has the highest base score, larger `alpha(n)` should sharpen attention onto the target at long lengths
- target attention mass should decay less quickly with length
- exactly-one positive accuracy should improve at long lengths

Main risk:

- this scales every score gap, not just target-vs-non-target gaps
- if a non-target key has the highest base score, the multiplier can make attention more confidently wrong
- if `k` grows too large, attention may become brittle or numerically saturated

### Hypothesis B: Target-Key Log Bias Is More Direct But More Task-Specific

Add a learned key-dependent bias that grows with `log(n)`:

```text
base_score(i, j) = q_i dot k_j / sqrt(d_head)
target_like_j = detector(key_representation_j)
beta(n) = softplus(k_bias) * log(n)
score(i, j) = base_score(i, j) + beta(n) * target_like_j
```

`target_like_j` must be learned from model representations. It must not use the true target mask or label.

Expected behavior:

- if the model learns a reliable target-key detector, the target key receives an explicit length-scaled boost
- target attention mass should remain more stable than in Stage 1
- long exactly-one positive logits should stop drifting below zero

Main risk:

- this introduces a stronger task-specific inductive bias than a standard transformer
- if the detector assigns high bias to non-target tokens, long-length behavior can still fail
- negative examples may become false positives if the detector is poorly calibrated

## Important Training Caveat

If the correction is parameterized as:

```text
alpha(n) = 1 + softplus(k) * log(n / train_length)
```

then `alpha(10) = 1` for length-10 training, and `k` receives no useful gradient under fixed length-10 training. Do not use this as the primary parameterization unless training includes multiple lengths.

For the primary length-10-only intervention, use a parameterization where `k` affects length 10 as well:

```text
1 + softplus(k) * log(n)
```

or:

```text
1 + softplus(k) * log1p(n)
```

Use careful initialization so the initial model is close to the Stage 1 transformer.

Recommended initialization:

```text
k_init = -5.0
softplus(k_init) ~= 0.0067
```

This gives a small initial correction while still allowing gradients at length 10.

## Model Variants

Keep the Stage 1 transformer backbone as close as possible:

- 1 transformer encoder layer
- 1 attention head initially
- `d_model = 64`
- no positional encoding
- max pooling
- binary classifier
- train length 10 for the primary experiment

Implement these variants:

### Variant 0: Stage 1 Baseline

Original attention:

```text
score(i, j) = q_i dot k_j / sqrt(d_head)
```

Purpose:

- reference point for accuracy, target attention decay, max-pool contribution drift, and positive logit collapse

### Variant 1: Global Log-Temperature Attention

Attention logits:

```text
score(i, j) = alpha(n) * q_i dot k_j / sqrt(d_head)
alpha(n) = 1 + softplus(k_temperature) * log1p(n)
```

Track:

- learned `k_temperature`
- `alpha(n)` for every evaluation length
- attention entropy by length
- target attention mass by length
- long-length positive and negative accuracy

### Variant 2: Target-Key Log-Bias Attention

Attention logits:

```text
target_like_j = detector(k_j)
score(i, j) = q_i dot k_j / sqrt(d_head) + beta(n) * target_like_j
beta(n) = softplus(k_bias) * log1p(n)
```

Possible detector:

```text
target_like_j = linear(k_j)
```

or:

```text
target_like_j = MLP(k_j)
```

Start with a single linear detector to keep the interpretation simple.

Track:

- learned `k_bias`
- `beta(n)` for every evaluation length
- `target_like_j` for target and non-target vocabulary tokens
- whether target-like scores separate the target token from non-target tokens
- target attention mass by length
- long-length positive and negative accuracy

## Training Conditions

Run each intervention under two training conditions.

### Condition A: Fixed Length 10

Train only on length 10, matching Stage 1.

Purpose:

- test whether the architecture alone can learn a useful length-aware correction from the original training distribution
- this is the strictest comparison to Stage 1

Expected difficulty:

- the model may not learn the correct extrapolating value of `k` from length 10 alone
- if this fails, it does not fully refute the intervention; it may indicate insufficient training signal

### Condition B: Short Multi-Length Calibration

Train on:

```text
[10, 20, 50, 100]
```

Purpose:

- test whether a small amount of length variation lets the model learn a better length-scaling correction
- compare against Stage 2A to see whether the improvement comes from multi-length training alone or from the length-aware attention mechanism

This condition should use the same data-loader strategy as Stage 2A: single-length batches, alternating across length-specific loaders, no padding, and no masks.

## Evaluation

Evaluate every run on the same primary and diagnostic slices used in previous stages.

Required lengths:

```text
10, 20, 50, 100, 200, 500, 700, 900, 1000, 1500, 2000, 5000, 10000
```

Required diagnostic slices:

- positive exactly one target
- positive multi-target `k=2`
- positive multi-target `k=3`
- positive multi-target `k=5`
- positive target near beginning
- positive target near middle
- positive target near end
- negative no target

Primary metrics:

- accuracy by diagnostic slice and length
- positive exactly-one accuracy by length
- negative accuracy by length
- mean positive logit by length
- mean negative logit by length
- logit margin:

```text
mean_positive_logit - mean_negative_logit
```

Mechanistic metrics:

- learned length multiplier values: `alpha(n)` or `beta(n)`
- attention entropy by length
- target attention mean and max by length
- softmax denominator mean by length
- target score mean and target score rank by length
- pooled activation norms by length
- max-pool target-sourced vs non-target-sourced contribution
- final logit decomposition by length

## Success Criteria

A length-aware attention intervention is useful if it satisfies all of the following:

- exactly-one positive accuracy remains high at lengths where Stage 1 fails
- negative accuracy remains high, so the model is not solving the task by predicting positive too often
- target attention mass decays less than in Stage 1
- positive logits do not drift below zero at long lengths
- the improvement is visible on sparse exactly-one positives, not only on multi-target positives
- the learned `k` parameter is non-trivial and contributes to the length-dependent behavior

The strongest result would be:

```text
Stage 1 fails at long exactly-one positives,
Stage 2A improves but still has some degradation,
log-aware attention further stabilizes target attention and logits.
```

## Failure Criteria

The intervention should be considered unsuccessful if:

- long exactly-one positives still collapse below zero
- negative examples become false positives at long lengths
- attention sharpens onto the wrong non-target key
- learned `k` remains effectively zero
- performance improves only because multi-target positives are easy
- the model becomes numerically unstable or produces NaNs at long lengths

## Implementation Plan

### Step 1: Add Length-Aware Attention Layers

Add model components that can compute attention logits manually instead of relying entirely on `nn.MultiheadAttention`.

Required behavior:

- support `batch_first=True`
- return attention weights for audit
- support one head first
- keep output projection, residual connections, layer norms, and feedforward block matching Stage 1
- expose learned length-scaling parameters for logging

Recommended files:

```text
src/models.py
src/attention.py
```

### Step 2: Add Configs

Add a Stage 2B config with fields such as:

```text
attention_variant: "global_log_temperature" | "target_key_log_bias"
log_length_mode: "log1p_length"
log_scale_init: -5.0
target_detector: "linear"
train_lengths: optional tuple for multi-length condition
```

Add example YAML configs:

```text
configs/stage2b_global_log_temperature.yaml
configs/stage2b_target_key_log_bias.yaml
configs/stage2b_global_log_temperature_multilength.yaml
configs/stage2b_target_key_log_bias_multilength.yaml
```

### Step 3: Add Training Script

Create:

```text
src/stage2b_length_aware_attention.py
```

It should support:

- fixed length-10 training
- multi-length calibration training
- diagnostic slices
- audit CSV examples
- attention summaries
- learned length-scale logging
- checkpoint and config metadata

Recommended output directories:

```text
runs/stage2b_global_log_temperature
runs/stage2b_target_key_log_bias
runs/stage2b_global_log_temperature_multilength
runs/stage2b_target_key_log_bias_multilength
```

### Step 4: Extend Attention Analysis

Update attention summaries to include:

- `attention_variant`
- `alpha_length_scale` or `beta_length_scale`
- raw base attention score statistics
- corrected attention score statistics
- target-like detector statistics for the target-key bias variant

For target-key bias, record vocabulary-level detector outputs:

```text
token_id, is_target, target_like_score
```

### Step 5: Tests

Add unit tests for:

- length scale is positive and changes with length
- length scale parameter receives gradient at length 10
- attention probabilities sum to 1 over keys
- output shapes match Stage 1
- target-key bias does not use oracle target masks
- forward pass has no NaNs for long lengths
- fixed-length and multi-length training runners can execute smoke tests

## Analysis Plan

After training, compare these runs:

1. Stage 1 baseline
2. Stage 2A multi-length transformer
3. Stage 2B global log-temperature, fixed length 10
4. Stage 2B target-key log-bias, fixed length 10
5. Stage 2B global log-temperature, multi-length calibration
6. Stage 2B target-key log-bias, multi-length calibration

For each run, answer:

- Did exactly-one positive accuracy remain stable at long lengths?
- Did negative accuracy remain stable?
- Did target attention mass decay more slowly than Stage 1?
- Did the learned length scale become meaningful?
- Did the intervention reduce harmful non-target max-pool contributions?
- Did the final positive logit remain above zero?

## Expected Interpretation

If global log-temperature works:

- the Stage 1 failure is likely caused largely by insufficient attention sharpness under denominator growth
- the target key already has enough rank advantage, but the softmax needs a length-dependent sharpening factor

If global log-temperature fails but target-key bias works:

- the model needs a target-specific length correction, not just sharper attention
- the target key is not consistently the top key under the base attention scores

If both fail under fixed length 10 but improve under multi-length calibration:

- the architecture can use a `log(n)` correction, but length-10-only training does not identify the correct extrapolating parameter

If both fail even under multi-length calibration:

- the Stage 1 degradation is not solved by attention scaling alone
- the max-pool classifier or another part of the architecture may be the dominant failure source
- the next priority should be final-classifier interventions, especially token-wise detector plus position-wise max

## Deliverables

- Stage 2B implementation
- Stage 2B YAML configs
- smoke tests and unit tests
- metrics CSVs for all four intervention runs
- attention summary CSVs including learned length-scale values
- short `EXPERIMENT.md` update comparing Stage 1, Stage 2A, and Stage 2B
- optional figures showing:
  - learned `alpha(n)` or `beta(n)`
  - target attention by length
  - exactly-one positive accuracy by length
  - positive and negative logits by length
  - max-pool contribution decomposition by length

## Immediate Next Step

Implement the global log-temperature variant first because it is the smallest architectural change.

Then implement the target-key log-bias variant and compare whether target-specific correction is necessary.
