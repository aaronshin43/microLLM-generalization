# Task: Stage 4B Ablation 3C - Inference-Time Top-K Readout

## Objective

Professor suggestion:

```text
Train the model normally, without top-k, on a fixed length such as n = 10.
Then, when using the same model to process larger n, use the top-k formula.
Maybe take top_k = 10, matching the trained length, though smaller values may also work.
```

This task tests that idea directly.

The central question is:

**Can a full-softmax-trained Stage 4B model length-generalize better if only the evaluation
readout replaces the full-sequence softmax denominator with a fixed top-k denominator?**

This is different from the previous warm-start experiment. Warm-start top-k fine-tuned a new
`topk_softmax_mass` model after loading score-side parameters from a full-softmax checkpoint.
That still put hard top-k inside the training objective, so ranking and readout calibration
could be destabilized during fine-tuning.

This task should not fine-tune with top-k. Instead:

```text
training:
    use the existing normally trained softmax_mass model

evaluation:
    keep the same learned score-side parameters and classifier
    replace only the readout denominator with top-k restricted softmax
```

## Background

The strict Stage 4B baseline trains with full softmax mass:

```math
m_k(n)
=
\frac{k e^{\alpha\Delta}}{k e^{\alpha\Delta} + (n-k)}.
```

At train length `n = 10`, the classifier learns thresholds for the readout distribution seen
under that denominator. At long length, the denominator changes because $(n-k)$ grows, so the
same classifier sees a shifted target-mass scale and positive counts collapse.

The inference-time top-k idea keeps the learned model fixed but changes the long-length
readout to:

```math
m_k^{\text{top-}R}
=
\frac{k e^{\alpha\Delta}}{k e^{\alpha\Delta} + (R-k)}
```

when all target positions enter the selected top-$R$ subset. If `R = 10`, then at the training
length this readout is exactly the same as full softmax because the top-10 subset contains the
whole length-10 sequence. At longer lengths, `R = 10` preserves the denominator scale the
classifier was calibrated on during training.

Smaller `R` values, such as `R = 3`, are also worth checking, but they are less conservative:
they change the readout even at `n = 10`, so any accuracy change mixes length correction with
classifier calibration shift.

## Comparison With Warm-Start Top-K

The previous warm-start method:

```text
1. train or reuse full-softmax checkpoint
2. copy only score-side parameters
3. reinitialize classifier
4. fine-tune with topk_softmax_mass
```

This task:

```text
1. train or reuse full-softmax checkpoint
2. copy/load the whole model, including classifier
3. do not train further
4. evaluate with a top-k readout override
```

Expected advantage:

```text
No cold-start hard-selection problem during training, because hard top-k is never part of the
training objective.
```

Main remaining risks:

```text
ranking failure:
    target positions still may not enter the top-k subset at long length

readout calibration shift:
    top_k smaller than the training length may produce a readout distribution the classifier
    was not trained on

positive saturation:
    length-aware multipliers may make all positive counts look too similar even with a fixed
    denominator
```

## Scope

Implement only the inference-time top-k diagnostic.

Keep unchanged:

- Stage 4B dataset and count labels,
- existing trained `softmax_mass` checkpoints,
- classifier weights from the source checkpoint,
- `constant`, `log`, and `learned_log` alpha modes,
- chunked and stratified evaluation,
- existing top-k diagnostics.

Do not implement yet:

- top-k fine-tuning,
- differentiable top-k relaxation,
- ranking auxiliary loss,
- learned or adaptive `top_k`,
- multi-length training,
- multi-target-type counting,
- full-transformer Stage 1/2 changes.

## Required Implementation

Add an evaluation-only path for Stage 4B checkpoints.

A suggested CLI/API is:

```text
--eval-only-checkpoint PATH
--eval-readout-mode topk_softmax_mass
--top-k 10
```

Behavior:

```text
load the full checkpoint model, including classifier
override only the evaluation readout mode and top_k
run evaluation without optimizer steps
write metrics to the requested output directory
```

The source checkpoint should remain a normally trained `softmax_mass` checkpoint. For the main
diagnostic, reuse the e500 baseline checkpoints:

```text
runs/stage4b/constant_e500_t1_nt1_k3/model.pt
runs/stage4b/log_e500_t1_nt1_k3/model.pt
runs/stage4b/learned_log_e500_t1_nt1_k3/model.pt
```

Validation requirements:

- checkpoint metadata must match the requested `target_token_count`, `non_target_token_count`,
  `max_target_count`, `d_head`, and `alpha_mode`,
- classifier weights must be loaded from the checkpoint,
- no training history should be produced unless it is clearly marked as source metadata,
- save both source checkpoint metadata and evaluation override metadata in `config.json`,
  `model.pt` if a model is written, and CSV outputs.

Recommended metadata fields:

```text
eval_only_checkpoint
train_readout_mode
eval_readout_mode
top_k
effective_top_k
```

## Diagnostics

Keep the current Stage 4B top-k diagnostics:

```text
mean_topk_target_count
mean_topk_target_recall
mean_topk_all_targets_included
mean_topk_non_target_count
mean_topk_target_mass
mean_topk_non_target_mass
mean_target_attention_mass
mean_min_margin_delta
worst_min_margin_delta
readout_finite_fraction
```

Key diagnostic interpretation:

```text
If top_k = 10 preserves train-length accuracy and improves 10M:
    denominator mismatch was a major cause of the baseline failure

If top_k = 10 preserves target inclusion but still fails:
    readout saturation or classifier calibration is still insufficient

If top_k = 10 excludes targets:
    the full-softmax-trained score ranking is not enough at long length

If top_k < 10 works better than top_k = 10:
    smaller denominators may provide a stronger count signal, but calibration shift must be
    reported explicitly
```

## Implementation Plan

### Step 1: Add Eval-Only Checkpoint Loading

Load a Stage 4B checkpoint and restore the full model state, including the classifier.

### Step 2: Add Evaluation Readout Override

Allow evaluation to use a readout mode different from the training checkpoint readout mode.
The primary override is:

```text
eval_readout_mode = topk_softmax_mass
```

The source checkpoint should remain:

```text
train_readout_mode = softmax_mass
```

### Step 3: Thread Metadata

Thread eval-only metadata through:

```text
config JSON
metrics CSVs
count metrics
confusion outputs
target-type metrics if present
```

### Step 4: Tests

Add or update tests under `infinite_generalization/tests/`:

1. Eval-only loading restores classifier weights from the checkpoint.
2. Eval readout override changes the readout mode without changing checkpoint weights.
3. `top_k = train_length` matches full-softmax readout on length-10 examples.
4. Metadata mismatch raises a clear error.
5. Chunked eval-only evaluation works for `topk_softmax_mass`.
6. A tiny eval-only run completes end to end.

### Step 5: Smoke Runs

Run tiny smoke configurations for:

```text
constant
log
learned_log
```

Use:

```text
eval_readout_mode = topk_softmax_mass
top_k             = train_length
```

Output directory:

```text
runs/stage4b/ablation3c_inference_topk_smoke/
```

### Step 6: Main Runs

Run the base Stage 4B setting by evaluating the existing e500 `softmax_mass` checkpoints with
inference-time top-k.

Primary setting:

```text
top_k = 10
```

Output directory:

```text
runs/stage4b/ablation3c_inference_topk/
```

Recommended run names:

```text
constant_e500_eval_topk10_t1_nt1_k3
log_e500_eval_topk10_t1_nt1_k3
learned_log_e500_eval_topk10_t1_nt1_k3
```

Use the same long evaluation setup as the prior Stage 4B runs:

```text
test_examples       = 720
eval_chunk_examples = 36
eval_lengths        = 10 ... 10000000
```

If the primary result is promising or ambiguous, add sensitivity checks without overwriting
the primary runs:

```text
top_k = 3
top_k = 4
top_k = 6
```

Recommended sensitivity directory:

```text
runs/stage4b/ablation3c_inference_topk/sensitivity/
```

### Step 7: Report Update

Update `STAGE4B_COUNTING_TARGET_OCCURRENCES.md` with a short section comparing:

```text
cold-start top-k training
warm-start top-k fine-tuning
inference-time top-k readout
```

## Analysis Questions

After the runs complete, answer:

```text
Does inference-time top_k = 10 preserve train-length accuracy?
Does it improve 10M accuracy over the strict full-softmax baseline?
Do target positions enter the top-k subset at long length?
Does the classifier trained on full-softmax mass remain calibrated under top-k readout?
Do smaller top_k values help or hurt relative to top_k = 10?
Does constant scaling benefit more than log or learned_log?
Does this support the hypothesis that the main issue was denominator growth rather than
top-k trainability?
```

## Done Criteria

This task is complete when:

1. Stage 4B supports eval-only loading from a full checkpoint.
2. Evaluation can override `softmax_mass` with `topk_softmax_mass` without retraining.
3. Tests cover eval-only loading, readout override, metadata validation, and chunked eval.
4. Smoke runs pass for all three alpha modes.
5. Main `top_k = 10` eval-only runs complete under
   `runs/stage4b/ablation3c_inference_topk/`.
6. The Stage 4B report is updated with the inference-time top-k result.

## Out Of Scope

- Training or fine-tuning with hard top-k.
- Reinitializing the classifier.
- Differentiable top-k relaxation.
- Ranking auxiliary loss.
- Learned or adaptive `top_k`.
- Multi-length training.
- Multi-target-type counting.
- Full-transformer Stage 1/2 changes.