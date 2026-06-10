# Task: Stage 3B Multi-Length Training In The Simplified Attention Model

## Objective

Extend the Stage 3 simplified length-aware attention experiment from single-length training to multi-length training.

Stage 3 showed that the reduced attention model can satisfy the two-score assumption and that learned-log attention can reach the asymptotic regime when optimization pushes:

```math
c\Delta>1.
```

Stage 3B asks whether training on several short lengths makes this easier or more stable.

The central question is:

**Does multi-length training help the reduced learned-log model reach the $c\Delta>1$ regime faster or more reliably than length-10-only training?**

## Motivation

The June 9 meeting notes suggested repeating the same simplified experiment while training simultaneously on several lengths, such as:

```text
10, 20, 50
```

This is the most direct next step because:

- it stays close to the professor's simplified model
- it keeps the two-token setup interpretable
- it tests a training-distribution change before adding architectural complexity
- it connects naturally to the earlier Stage 2A/2B multi-length experiments

The key hypothesis is:

**Multi-length training may give the optimizer stronger pressure to increase the effective learned-log exponent $c\Delta$, because the model must solve the task under more than one softmax denominator size during training.**

## Scope

This task modifies the existing Stage 3 reduced-model pipeline.

Primary implementation file:

```text
infinite_generalization/src/stage3_simplified_attention.py
```

Primary report to update or extend after running:

```text
infinite_generalization/documents/STAGE3_SIMPLIFIED_LENGTH_AWARE_ATTENTION.md
```

Recommended run output directories:

```text
runs/stage3b_single_length_learned_log
runs/stage3b_multilength_learned_log_10_20_50
runs/stage3b_multilength_learned_log_10_20_50_100
```

Do not add full transformer components in this task. Stage 3B should remain a reduced-model experiment.

## Experimental Design

### Model

Use the same reduced model as Stage 3:

- two tokens: target $t=0$ and non-target $u=1$
- fixed one-hot values: $t\mapsto[1,0]$, $u\mapsto[0,1]$
- learned query projection $W_Q$
- learned key projection $W_K$
- values equal to one-hot inputs
- last-query attention
- linear classifier on the attention output

Keep the target at position 0 for positives:

```text
t, u, u, ..., u
```

Keep negatives as:

```text
u, u, u, ..., u
```

This preserves the exact Stage 3 theory setup and keeps the two-score assumption easy to audit.

### Training Conditions

Compare at least two training conditions.

#### Condition A: Single-Length Baseline

Train only at length 10:

```text
train_lengths = [10]
```

This reproduces the Stage 3 setup.

#### Condition B: Multi-Length Training

Train on several short lengths:

```text
train_lengths = [10, 20, 50]
```

Optional additional condition:

```text
train_lengths = [10, 20, 50, 100]
```

The first implementation should support arbitrary training-length lists through a command-line option:

```text
--train-lengths 10 20 50
```

### Alpha Modes

The primary mode for Stage 3B is:

```text
learned_log
```

because the main question is whether training reaches:

```math
c\Delta>1.
```

Optional controls:

- `constant`: should still fail asymptotically because $\Delta$ remains fixed.
- `log`: should succeed if $\Delta>1$.

These controls are useful but secondary. Do not let them delay the primary learned-log comparison.

### Fairness: Match Optimizer Updates

Multi-length training can change the number of optimizer updates per epoch. Therefore, comparisons should not rely only on epoch count.

Add or use a way to control the total optimizer update budget.

Recommended option:

```text
--max-train-steps 6400
```

Expected behavior:

- if `--max-train-steps` is provided, stop training after that many optimizer updates
- if it is not provided, keep the existing epoch-based behavior

This allows fair comparisons such as:

```text
single length, 6400 updates
multi length, 6400 updates
```

If adding `--max-train-steps` is too disruptive, document the exact number of optimizer updates for every run and compare runs with similar update counts.

## Training Data Construction

For each training length, build a balanced dataset:

- half positive examples
- half negative examples

For multi-length training, use one of these approaches.

Recommended approach:

**Concatenate balanced datasets from each training length and shuffle the combined dataset.**

This is simple and makes each training length equally represented per epoch.

Alternative approach:

**Cycle separate dataloaders by length until all requested optimizer steps are consumed.**

This gives more explicit length control but is more code.

For the first Stage 3B implementation, concatenation is sufficient.

## Evaluation Lengths

Evaluate on the same long sweep used in the latest Stage 3 analysis:

```text
10, 100, 1000, 10000, 100000, 1000000, 5000000, 10000000
```

Keep evaluation examples small enough for long lengths to be practical.

Recommended defaults:

```text
test_examples = 50
eval_batch_size = 16
```

These are sufficient for mechanism-level metrics because this reduced dataset is deterministic and highly controlled.

## Metrics To Save

Keep all current Stage 3 metrics.

Required columns in `metrics_by_length.csv`:

- `length`
- `split`
- `alpha_mode`
- `alpha_value`
- `accuracy`
- `positive_accuracy`
- `negative_accuracy`
- `mean_logit_positive`
- `mean_logit_negative`
- `mean_probability_positive`
- `mean_probability_negative`
- `mean_target_score_a`
- `mean_non_target_score_b`
- `mean_delta`
- `std_non_target_scores`
- `mean_empirical_target_attention`
- `mean_theory_target_attention_using_empirical_delta`
- `mean_theory_target_attention_using_train_delta`
- `mean_attention_absolute_error_empirical_vs_theory`
- `learned_alpha_coefficient`
- `classifier_weight_target_coord`
- `classifier_weight_non_target_coord`
- `classifier_bias`

Add these Stage 3B metadata columns if practical:

- `train_lengths`
- `train_length_count`
- `optimizer_updates`
- `examples_per_train_length`

If adding them to every row is inconvenient, save them clearly in `config.json`.

## Key Derived Quantities

For learned-log runs, compute and report:

```math
c\Delta.
```

Use:

```text
c = learned_alpha_coefficient
Delta = mean_delta
```

The most important comparison is:

| Condition | Updates | $c$ | $\Delta$ | $c\Delta$ | Positive logit at 10M | Positive accuracy at 10M |
|---|---:|---:|---:|---:|---:|---:|
| single length | TBD | TBD | TBD | TBD | TBD | TBD |
| multi length | TBD | TBD | TBD | TBD | TBD | TBD |

## Main Questions To Answer

### 1. Does Multi-Length Training Increase $c\Delta$?

Compare:

- learned $c$
- learned $\Delta$
- product $c\Delta$

The key threshold is:

```math
c\Delta>1.
```

### 2. Does Multi-Length Training Reach The Threshold With Fewer Updates?

Compare update-matched runs.

Example:

- single-length learned-log at 1600, 3200, 6400 updates
- multi-length learned-log at 1600, 3200, 6400 updates

The strongest result would be:

**Multi-length training crosses $c\Delta>1$ with fewer optimizer updates than single-length training.**

### 3. Does The Two-Score Assumption Still Hold?

Check:

```math
\operatorname{std}_{j\ne t}(s_j)=0.
```

Expected result:

It should remain exactly 0.0 because all non-target tokens are identical and use fixed one-hot values.

If it does not remain zero, inspect the implementation.

### 4. Does The Closed-Form Attention Formula Still Match?

This is a secondary check.

Once the two-score assumption holds, empirical target attention should match:

```math
p_t(n)
=
\frac{e^{\alpha\Delta}}
{e^{\alpha\Delta}+(n-1)}.
```

The more important question is whether training reaches a better $c\Delta$ regime.

### 5. Does Positive Logit Stay Stable At Very Long Lengths?

Track:

- target attention mass
- positive logit
- positive accuracy

The model may have high target attention but still fail if the final classifier is poorly calibrated. Save classifier weights to diagnose this.

## Implementation Steps

### Step 1: Add Multi-Length Training Support

Modify `stage3_simplified_attention.py` to accept:

```text
--train-lengths 10 20 50
```

Behavior:

- if `--train-lengths` is omitted, default to the existing single `train_length`
- if provided, use the list as the training lengths
- keep backward compatibility with `--train-length`

### Step 2: Build Multi-Length Training Data

For each training length:

1. generate balanced examples
2. concatenate all examples
3. shuffle deterministically
4. train with the existing DataLoader path

Save the training lengths in `config.json`.

### Step 3: Add Optional Update-Budget Control

Add:

```text
--max-train-steps
```

Behavior:

- train until the step budget is reached
- allow the dataloader to restart across epochs if needed
- save the actual number of optimizer updates

This makes single-length and multi-length runs comparable.

### Step 4: Run A Smoke Test

Run a tiny multi-length smoke test:

```text
$env:PYTHONPATH = 'src'; ..\.venv\Scripts\python.exe src\stage3_simplified_attention.py --smoke-test --alpha-mode learned_log --train-lengths 10 20
```

Expected:

- script exits successfully
- `metrics_by_length.csv` is created
- `config.json` records multiple train lengths
- no NaN appears in key metrics

### Step 5: Run Primary Experiments

Recommended primary runs:

```text
$env:PYTHONPATH = 'src'; ..\.venv\Scripts\python.exe src\stage3_simplified_attention.py --alpha-mode learned_log --train-lengths 10 --max-train-steps 1600 --output-dir runs/stage3b_single_length_learned_log_steps1600
```

```text
$env:PYTHONPATH = 'src'; ..\.venv\Scripts\python.exe src\stage3_simplified_attention.py --alpha-mode learned_log --train-lengths 10 20 50 --max-train-steps 1600 --output-dir runs/stage3b_multilength_learned_log_10_20_50_steps1600
```

Repeat for:

```text
3200, 6400
```

Optional extended run:

```text
12800
```

### Step 6: Analyze Results

Create or update a Stage 3B analysis section in:

```text
documents/STAGE3_SIMPLIFIED_LENGTH_AWARE_ATTENTION.md
```

The analysis should answer:

- Does multi-length training increase $c$?
- Does it increase $\Delta$?
- Does it increase $c\Delta$?
- Does it cross $c\Delta>1$ earlier?
- Does it improve positive logit at 10M?
- Does it preserve the two-score assumption?

## Expected Outcomes

### Outcome A: Multi-Length Helps

If multi-length training reaches $c\Delta>1$ with fewer updates, then:

**Multi-length training improves optimization toward the asymptotic regime in the reduced model.**

This would support trying analogous length-diverse training in fuller transformer settings.

### Outcome B: Multi-Length Does Not Help

If multi-length and single-length training have similar $c\Delta$ at matched update budgets, then:

**The reduced model may already get enough length information from length 10, and optimization strength may matter more than length diversity.**

This would suggest focusing on architecture or classifier changes rather than only training distribution.

### Outcome C: Multi-Length Hurts

If multi-length training reduces $c\Delta$ or hurts long-length logits, then:

**The mixed training distribution may make optimization harder or change classifier calibration.**

Inspect:

- learned $c$
- learned $\Delta$
- classifier weights
- target attention mass by length
- positive logits by length

## Success Criteria

This task is complete when:

- `stage3_simplified_attention.py` supports multi-length training
- smoke test passes for at least two training lengths
- update-matched single-length and multi-length learned-log runs are completed
- metrics include enough information to compute $c\Delta$
- the report states whether multi-length training helps reach $c\Delta>1$
- the conclusion does not overclaim beyond the reduced-model setting

## Risks And Notes

- Do not interpret finite-length success as infinite-length success unless $c\Delta>1$.
- Keep the target at position 0 for this task; target-anywhere is a separate follow-up.
- Keep fixed one-hot values; learned embeddings are a separate follow-up.
- If multi-length training changes the number of updates per epoch, use update-matched comparisons.
- The result applies only to the reduced model unless later experiments show the full transformer preserves an analogous score pattern.
