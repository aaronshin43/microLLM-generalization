# Task: Stage 3C Target Can Appear Anywhere

## Objective

Extend the Stage 3 reduced attention model so that the target token can appear at any non-final position.

Stage 3 used positive inputs of the form:

```text
t, u, u, ..., u
```

This fixed the target at position 0. Stage 3C removes that positional simplification while preserving the most important theoretical condition:

```text
the final readout query should still be produced by the non-target token u.
```

Therefore, for a sequence of length $n$, positive examples should place the single target token at:

```math
p \in \{0,1,\ldots,n-2\}.
```

The last token must remain:

```text
u
```

so that:

```math
q_{\mathrm{last}}=q_u.
```

The central question is:

**Does the reduced model learn a position-independent target detector when the target can appear anywhere except the final readout position?**

## Motivation

This is the second next-step experiment from the June 9 meeting notes:

```text
Extend theory and empirical results to the case where target can be anywhere.
```

The goal is to test whether the Stage 3 mechanism depends on the target always appearing at position 0, or whether it genuinely detects the target token identity independent of position.

If the model is position-independent, then the score row for a positive example with target at position $p$ should look like:

```math
S_n=(b,\ldots,b,a,b,\ldots,b),
```

where:

- $a$ is the score assigned to the target key.
- $b$ is the shared score assigned to each non-target key.
- the location of $a$ changes with target position $p$.

The closed-form target attention formula should remain:

```math
p_t(n)
=
\frac{e^{\alpha\Delta}}
{e^{\alpha\Delta}+(n-1)}.
```

Here:

```math
\Delta=a-b.
```

## Scope

Modify the existing Stage 3 reduced-model pipeline.

Primary implementation file:

```text
infinite_generalization/src/stage3_simplified_attention.py
```

Primary report to update after running:

```text
infinite_generalization/documents/STAGE3_SIMPLIFIED_LENGTH_AWARE_ATTENTION.md
```

Do not add full transformer components in this task.

Do not add positional encoding.

Do not allow the target at the final position in the first Stage 3C version.

## Experimental Design

### Model

Use the same reduced model as Stage 3:

- two tokens: target $t=0$ and non-target $u=1$
- fixed one-hot values: $t\mapsto[1,0]$, $u\mapsto[0,1]$
- learned query projection matrix
- learned key projection matrix
- fixed values equal to the one-hot inputs
- last-query attention
- linear classifier on the attention output

### Positive Inputs

For length $n$, positive examples should contain exactly one target:

```text
u, ..., t, ..., u
```

with:

```math
target\_position \in \{0,1,\ldots,n-2\}.
```

The final token must always be:

```text
u
```

### Negative Inputs

Negative examples remain:

```text
u, u, u, ..., u
```

### Why The Final Position Is Excluded

If the target appears at the final position, then the final query becomes:

```math
q_{\mathrm{last}}=q_t.
```

That changes the theoretical setup. The current Stage 3 model assumes:

```math
q_{\mathrm{last}}=q_u.
```

Therefore Stage 3C should first vary target position while keeping the readout query fixed as the non-target query.

## Conditions To Run

Run the original Stage 3 conditions, but with target-anywhere positives.

All runs should use:

```text
train_lengths = [10]
target_position_mode = nonfinal_random
eval_lengths = 10, 100, 1000, 10000, 100000, 1000000, 5000000, 10000000
test_examples = 50
eval_batch_size = 8
```

Use update budgets matching the earlier Stage 3 epoch-equivalent runs:

| Run | Alpha mode | Epoch equivalent | Max train steps |
|---|---|---:|---:|
| `constant_e50` | `constant` | 50 | 1600 |
| `constant_e100` | `constant` | 100 | 3200 |
| `constant_e1000` | `constant` | 1000 | 32000 |
| `log_e50` | `log` | 50 | 1600 |
| `learned_log_e50` | `learned_log` | 50 | 1600 |
| `learned_log_e100` | `learned_log` | 100 | 3200 |
| `learned_log_e200` | `learned_log` | 200 | 6400 |

Recommended output root:

```text
runs/stage3c_target_anywhere/
```

## Required Implementation Changes

### Step 1: Add Target Position Mode

Add a command-line option such as:

```text
--target-position-mode fixed_start
--target-position-mode nonfinal_random
```

Expected behavior:

- `fixed_start`: existing Stage 3 behavior, target at position 0.
- `nonfinal_random`: target sampled uniformly from positions `0` through `length - 2`.

Default should remain:

```text
fixed_start
```

to preserve backward compatibility.

### Step 2: Update Dataset Generation

Modify the two-token dataset generator so positive examples can use:

```text
target_position_mode = nonfinal_random
```

For each positive example:

1. start with all non-target tokens
2. sample a target position from `0` to `length - 2`
3. place the target token there
4. keep the final position as non-target

Negative examples remain all non-target.

### Step 3: Save Target Position Metadata

For evaluation, save enough information to audit target-position behavior.

At minimum, add aggregate metrics by target-position bucket:

- `beginning`
- `middle`
- `end_nonfinal`

Useful metrics:

- positive accuracy by bucket
- target attention by bucket
- mean delta by bucket
- non-target score std by bucket

If exact per-position metrics are cheap, save them as a separate CSV:

```text
target_position_metrics.csv
```

### Step 4: Preserve Existing Metrics

Keep all existing Stage 3 metrics:

- `accuracy`
- `positive_accuracy`
- `negative_accuracy`
- `mean_delta`
- `std_non_target_scores`
- `mean_empirical_target_attention`
- `mean_theory_target_attention_using_empirical_delta`
- `learned_alpha_coefficient`
- classifier weights

Add config metadata:

- `target_position_mode`
- whether final target positions are allowed

### Step 5: Smoke Test

Run a small smoke test:

```text
$env:PYTHONPATH = 'src'; ..\.venv\Scripts\python.exe src\stage3_simplified_attention.py --smoke-test --alpha-mode learned_log --target-position-mode nonfinal_random --output-dir runs/stage3c_target_anywhere/smoke
```

Expected:

- `model.pt` is created
- `metrics_by_length.csv` is created
- target-position metadata is saved
- final token remains non-target in positives
- no NaN appears in key metrics

### Step 6: Run Stage 3C Conditions

Run:

```text
constant_e50
constant_e100
constant_e1000
log_e50
learned_log_e50
learned_log_e100
learned_log_e200
```

under:

```text
runs/stage3c_target_anywhere/
```

Use the max-train-step budgets listed above.

### Step 7: Analyze Results

Answer:

1. Does the model still satisfy the two-score assumption?
2. Does $\Delta$ remain stable across target positions?
3. Does target attention remain stable across target positions?
4. Does learned-log e200 still reach $c\Delta>1$?
5. Does any target-position bucket fail earlier?
6. Does target-anywhere training change $c$, $\Delta$, or classifier calibration?

## Expected Outcomes

### Outcome A: Target Position Does Not Matter

If the model works equally well across target positions:

**The reduced model learned a position-independent token detector.**

Expected signs:

- non-target score std remains 0
- $\Delta$ is stable across target-position buckets
- target attention is stable across target-position buckets
- learned-log e200 still reaches $c\Delta>1$

### Outcome B: Target Position Matters

If performance varies by target position:

**The fixed-position Stage 3 result depended on more than token identity.**

Inspect:

- target attention by bucket
- mean delta by bucket
- positive logit by bucket
- whether positions near the final query behave differently

### Outcome C: Final-Adjacent Targets Behave Differently

If `end_nonfinal` behaves differently:

**The model may be sensitive to target proximity to the readout query, even without explicit positional encoding.**

This would be surprising in the reduced no-position model and should be checked carefully for implementation artifacts.

## Success Criteria

This task is complete when:

- target-anywhere dataset generation is implemented
- the final token remains non-target for positives
- Stage 3C smoke test passes
- all seven Stage 3C conditions are run
- target-position metrics are saved
- the report states whether the reduced model remains position-independent
- conclusions do not overclaim beyond the reduced no-position model

## Notes

- This task keeps exactly one target token in positives.
- This task keeps binary classification.
- This task keeps fixed one-hot values.
- This task keeps the final readout query as a non-target query.
- Allowing the target at the final position is a separate follow-up.
- Adding positional encodings is a separate follow-up.
