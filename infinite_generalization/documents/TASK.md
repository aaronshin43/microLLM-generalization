# Task: Stage 3E Multiple Target Token Types

## Objective

Stage 3E tests whether the reduced Stage 3 length-aware attention model can learn a **target class detector**, not just a detector for one specific target token.

Previous Stage 3 experiments used one target token type:

```text
target token: t
```

Stage 3E changes this to a target set:

```text
target tokens: t_1, t_2, ..., t_H
```

The binary task is unchanged:

```text
positive: the sequence contains one token from the target set
negative: the sequence contains no token from the target set
```

For the first Stage 3E pass, each positive example should contain exactly one target token, and each negative example should contain zero target tokens.

The central question is:

**Can the reduced model learn to detect a class of target tokens, and is long-length behavior controlled by the weakest target type?**

## Why This Comes Next

Stage 3D increased non-target-side complexity by adding multiple non-target token types.

Stage 3E increases target-side complexity by adding multiple target token types.

This is a more realistic existential task. Many real binary detection tasks ask whether a sequence contains any token from a target category, not only whether it contains one specific token.

## Controlled First Setup

The first Stage 3E experiment should isolate the new variable:

```text
multiple target token types
single non-target token type
target fixed at position 0
```

Use:

```text
target_token_count = 3
non_target_token_count = 1
target_position_mode = fixed_start
```

Do not combine Stage 3E with target-anywhere placement or multiple non-target token types in the first pass. Those are follow-up checks.

This keeps the theory simple and makes it easier to tell whether any failure comes from multiple target types rather than from position or non-target variation.

## Token Convention

Use contiguous token ids:

```text
target token ids: 0, 1, ..., H-1
non-target token ids: H, H+1, ..., H+M-1
```

For the first pass:

```text
H = 3
M = 1
```

So:

```text
target token ids: 0, 1, 2
non-target token id: 3
```

## Input Distribution

### Positive Inputs

Each positive example contains exactly one target token at position 0:

```text
t_h, u, u, ..., u
```

where $t_h$ is sampled uniformly from the target token set.

### Negative Inputs

Each negative example contains only the non-target token:

```text
u, u, u, ..., u
```

### Later Follow-Ups

After the controlled first pass, possible follow-ups are:

- multiple target types with multiple non-target types
- multiple target types with target-anywhere placement
- multiple target types with both target-anywhere placement and multiple non-target types

These should not be included in the first Stage 3E implementation unless the base result is already understood.

## Representation Design

As in Stage 3D, separate:

1. the representation used to compute query/key attention scores
2. the value representation averaged by attention

The score representation should distinguish all token ids:

```text
t_1, ..., t_H, u_1, ..., u_M
```

The attention value mapping should remain binary:

```math
t_h \mapsto [1,0],
\qquad
u_k \mapsto [0,1].
```

All target token types share the same value meaning: target evidence.

All non-target token types share the same value meaning: non-target evidence.

This keeps the attention output interpretable as:

```math
(p_{\mathrm{target}},1-p_{\mathrm{target}}).
```

For the first Stage 3E pass, positive examples contain exactly one target token, so $p_{\mathrm{target}}$ is simply the attention mass on that target position.

## Theory

In the controlled first setup, there is only one non-target token type and the final query is always the non-target query. Therefore, the score row for a positive example with target type $h$ is:

```math
S_n^{(h)}=(a_h,b,b,\ldots,b).
```

Variables:

- $h$ is the target token type.
- $a_h$ is the target key score for target type $h$.
- $b$ is the non-target key score.
- $\Delta_h=a_h-b$ is the target-vs-non-target margin for target type $h$.

The target attention mass is:

```math
p_t(n\mid h)
=
\frac{e^{\alpha a_h}}
{e^{\alpha a_h}+(n-1)e^{\alpha b}}
=
\frac{1}
{1+(n-1)e^{-\alpha\Delta_h}}.
```

The bottleneck is the target type with the smallest margin:

```math
\Delta_{\min}=\min_h \Delta_h.
```

For learned-log attention:

```math
\alpha=1+c\log(1+n),
```

the asymptotic diagnostic is:

```math
c\Delta_{\min}>1.
```

The key new question is whether all target token types get sufficiently large margins, or whether one weak target type controls long-length failure.

## Required Implementation Changes

Primary file:

```text
infinite_generalization/src/stage3_simplified_attention.py
```

### Step 1: Add Target Token Count

Add a CLI/config option:

```text
--target-token-count 1
```

Default:

```text
1
```

This preserves existing Stage 3 behavior.

### Step 2: Update Token Id Convention

When `target_token_count=H` and `non_target_token_count=M`:

```text
target ids = 0 ... H-1
non-target ids = H ... H+M-1
```

Existing Stage 3D behavior with one target token should remain equivalent when:

```text
target_token_count = 1
```

### Step 3: Update Dataset Generation

Required behavior:

- positives contain exactly one target token
- positive target token id is sampled uniformly from target ids
- negatives contain zero target tokens
- non-target positions are sampled from non-target ids
- `target_position_mode=fixed_start` places the sampled target at position 0
- `target_position_mode=nonfinal_random` should still work later, but is not required for the first Stage 3E run

The dataset should return both:

```text
target_positions
target_token_ids
```

For negative examples:

```text
target_position = -1
target_token_id = -1
```

### Step 4: Update Value Mapping

The fixed attention value mapping should be:

```math
\mathrm{value}(x)=
\begin{cases}
[1,0], & x \in \text{target ids} \\
[0,1], & x \in \text{non-target ids}.
\end{cases}
```

The classifier input should remain 2-dimensional.

### Step 5: Preserve Existing Metrics

Keep existing outputs:

```text
metrics_by_length.csv
non_target_type_metrics.csv
target_position_metrics.csv
```

Existing Stage 3, Stage 3C, and Stage 3D commands should continue to work with default:

```text
--target-token-count 1
```

### Step 6: Add Target-Type Metrics

Add:

```text
target_type_metrics.csv
```

Each row should include:

```text
length
split
alpha_mode
target_token_count
non_target_token_count
target_position_mode
final_query_token_id
target_token_id
positive_examples
positive_accuracy
mean_target_score
mean_target_attention
mean_min_margin
worst_observed_min_margin
mean_c_delta_min
worst_observed_c_delta_min
```

Important:

`final_query_token_id` should be included even though the first controlled setup has only one non-target token type. This keeps the metric schema compatible with later Stage 3E follow-ups that add multiple non-target token types or target-anywhere placement.

### Step 7: Add Aggregate Target-Class Diagnostics

In `metrics_by_length.csv`, add or preserve aggregate diagnostics:

```text
mean_min_margin_delta
min_margin_delta
learned_log_c_delta_min_mean
learned_log_c_delta_min_worst
```

For Stage 3E, these should aggregate over positive examples and therefore include target-type variation.

## Smoke Test

Run a small smoke test:

```powershell
$env:PYTHONPATH = 'src'; ..\.venv\Scripts\python.exe src\stage3_simplified_attention.py --smoke-test --alpha-mode learned_log --target-token-count 3 --non-target-token-count 1 --output-dir runs/stage3e_multiple_targets/smoke
```

Expected:

- `model.pt` is created
- `metrics_by_length.csv` is created
- `target_type_metrics.csv` is created
- positives contain exactly one target token
- negatives contain zero target tokens
- all target token ids appear in positive examples
- no unexpected NaN appears in key metrics
- default `target_token_count=1` still works

## Main Runs

Use the representative conditions:

```text
target_token_count = 3
non_target_token_count = 1
target_position_mode = fixed_start
train_lengths = [10]
test_examples = 50
eval_batch_size = 8
```

Output root:

```text
runs/stage3e_multiple_targets/
```

Run:

| Run | Alpha mode | Max train steps | Purpose |
|---|---|---:|---|
| `constant_e100_t3_nt1` | `constant` | 3200 | check whether constant multiplier still fails |
| `log_e50_t3_nt1` | `log` | 1600 | check whether fixed log succeeds when every target margin is above 1 |
| `learned_log_e200_t3_nt1` | `learned_log` | 6400 | check whether learned-log reaches $c\Delta_{\min}>1$ |

## Analysis Questions

### 1. Does The Model Learn A Target Class Detector?

Check whether all target token types have high positive accuracy near training length and at long lengths.

### 2. Is One Target Type Weaker?

Use:

```text
target_type_metrics.csv
```

Look for the target token type with the smallest margin:

```math
\arg\min_h \Delta_h.
```

### 3. Does Long-Length Behavior Follow The Smallest Target Margin?

For learned-log, check:

```math
c\min_h\Delta_h>1.
```

If one target type has $c\Delta_h<1$, it may become the first positive subtype to fail at long length.

### 4. Does Fixed Log Still Work?

For fixed log:

```math
\alpha=\log n.
```

Expected success condition:

```math
\Delta_{\min}>1.
```

### 5. Does Constant Multiplier Still Fail?

Expected:

Constant multiplier may fit training length but should still fail at sufficiently long length because fixed margins do not scale with $\log n$.

## Expected Outcomes

### Outcome A: Target Scores Collapse

All target types receive nearly identical target scores.

Interpretation:

The model learns a shared target-class representation at the attention-score level.

### Outcome B: Target Scores Differ But All Margins Are Large Enough

Target scores differ, but the smallest target margin remains sufficient.

Interpretation:

Exact target-score collapse is not necessary. The relevant diagnostic is the smallest target-vs-non-target margin.

### Outcome C: One Weak Target Type Controls Failure

One target type has a much smaller margin.

Interpretation:

The target class is only as robust as its weakest target type.

## Reporting Plan

Create a dedicated Stage 3E report if results are nontrivial:

```text
infinite_generalization/documents/STAGE3E_MULTIPLE_TARGET_TOKENS.md
```

The report should explain:

- why Stage 3E isolates target-side complexity
- whether the model learns a target class detector
- whether target token types collapse or remain distinct
- which target type has the smallest margin
- whether learned-log reaches $c\Delta_{\min}>1$

## Success Criteria

This task is complete when:

- `--target-token-count` is implemented
- existing Stage 3 behavior is preserved by default
- `target_type_metrics.csv` is produced
- smoke tests pass
- the three main Stage 3E runs complete
- target-type results are analyzed

## Expected Conclusion

The likely conclusion is:

**Stage 3E should preserve the same softmax dilution mechanism, but the bottleneck should move from a single target-vs-non-target margin to the target token type with the smallest margin.**
