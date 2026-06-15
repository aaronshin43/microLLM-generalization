# Task: Stage 3D Multiple Non-Target Tokens

## Objective

Stage 3D tests whether the reduced Stage 3 length-aware attention model can still generalize when there are multiple non-target token types.

The original Stage 3 theory used one target token and one non-target token. In that setting, a positive sequence produced a score row of the form:

```math
S_n=(a,b,b,\ldots,b).
```

Here:

- $a$ is the attention score assigned to the target key.
- $b$ is the shared attention score assigned to every non-target key.
- $n$ is the sequence length.

This is the two-score assumption. It makes the target attention mass reduce to:

```math
p_t(n)
=
\frac{e^{\alpha a}}
{e^{\alpha a}+(n-1)e^{\alpha b}}
=
\frac{1}
{1+(n-1)e^{-\alpha(a-b)}}.
```

Stage 3D breaks this simplification by adding several non-target token types. The score row can then become:

```math
S_n=(a,b_1,b_2,b_3,\ldots).
```

The denominator is no longer $(n-1)e^{\alpha b}$. The central question is:

**Can infinite-length generalization still work when non-target scores are not all identical?**

## Motivation

This task corresponds to item 5 in the June 9 meeting notes:

```text
Add additional non-target tokens.
Again, does the theory imply that infinite length generalization is possible?
```

This is important because realistic vocabularies do not contain only one non-target token. If the theory only works when all non-target keys have exactly the same score, then the Stage 3 result is fragile. If the theory extends to multiple non-target types, then the simplified model gives a stronger explanation of what length-aware attention needs to learn.

## Core Theory To Derive First

Assume:

- one target token type $t$
- $m$ non-target token types $u_1,\ldots,u_m$
- one target occurrence in a positive example
- $c_k(n)$ occurrences of non-target token $u_k$ in a length-$n$ sequence
- target score $a$
- non-target score $b_k$ for token type $u_k$
- attention multiplier $\alpha$

Then:

```math
\sum_{k=1}^{m} c_k(n)=n-1.
```

The target attention mass is:

```math
p_t(n)
=
\frac{e^{\alpha a}}
{e^{\alpha a}+\sum_{k=1}^{m} c_k(n)e^{\alpha b_k}}.
```

Divide numerator and denominator by $e^{\alpha a}$:

```math
p_t(n)
=
\frac{1}
{1+\sum_{k=1}^{m} c_k(n)e^{-\alpha(a-b_k)}}.
```

Define the per-token-type margin:

```math
\Delta_k=a-b_k.
```

Then:

```math
p_t(n)
=
\frac{1}
{1+\sum_{k=1}^{m} c_k(n)e^{-\alpha\Delta_k}}.
```

The dangerous term is the non-target type with the smallest positive margin:

```math
\Delta_{\min}=\min_k \Delta_k.
```

A small margin means that the corresponding non-target score $b_k$ is closest to the target score $a$. Its denominator term $c_k(n)e^{-\alpha\Delta_k}$ decays the slowest, so one hard non-target type can dominate the sum even when the other non-target types have much larger margins. Large-margin non-targets do not average away the smallest-margin bottleneck because the denominator depends exponentially on each margin.

If every non-target type appears with frequency proportional to $n$, then a sufficient long-length condition is:

```math
\alpha\Delta_{\min}-\log n \to +\infty.
```

For learned-log attention:

```math
\alpha=1+c\log(1+n),
```

the asymptotic condition becomes approximately:

```math
c\Delta_{\min}>1.
```

This is the Stage 3D analogue of the original Stage 3 condition $c\Delta>1$.

## Key Questions

1. Does the model collapse all non-target token types into one shared score?
2. If non-target scores differ, does the worst-case margin $\Delta_{\min}$ still become large enough?
3. Is long-length success controlled by $c\Delta_{\min}>1$?
4. Which non-target token type contributes most to the attention denominator?
5. Do constant, log, and learned-log attention behave differently when the two-score assumption is broken?

## Scope

Primary implementation file:

```text
infinite_generalization/src/stage3_simplified_attention.py
```

Primary report to update after running:

```text
infinite_generalization/documents/STAGE3_SIMPLIFIED_LENGTH_AWARE_ATTENTION.md
```

Optional dedicated report after the experiment:

```text
infinite_generalization/documents/STAGE3D_MULTIPLE_NON_TARGET_TOKENS.md
```

Do not add full transformer components in this task.

Do not add positional encoding.

Do not change the task from binary target-present classification.

## Experimental Design

### Token Convention

Use:

```text
target token id: 0
non-target token ids: 1, 2, ..., m
```

Add a CLI option:

```text
--non-target-token-count 1
```

Default:

```text
1
```

This preserves existing Stage 3 behavior.

Stage 3D should run with values such as:

```text
--non-target-token-count 2
--non-target-token-count 4
--non-target-token-count 8
```

### Score Representation vs Fixed Attention Value Mapping

Stage 3D should distinguish two different representations:

1. the representation used to compute attention scores
2. the representation averaged by the attention weights

In the original Stage 3 implementation, these were the same matrix. The fixed one-hot embedded matrix $X_n$ was used to compute query and key scores, and the same $X_n$ was also averaged by attention:

```math
Q=X_nW_Q,
\qquad
K=X_nW_K,
\qquad
\mathrm{attention\ output}=\mathrm{softmax}(\alpha QK^\top)X_n.
```

Therefore, in the two-token setup, the embedded matrix also acted as the attention value matrix:

```math
V_n=X_n.
```

For Stage 3D, this should be separated conceptually.

The score representation should distinguish token types:

```text
t, u_1, u_2, ..., u_m
```

This allows the model to assign different key scores:

```math
a,b_1,b_2,\ldots,b_m.
```

However, the fixed attention value mapping should remain binary:

```math
t \mapsto [1,0],
\qquad
u_k \mapsto [0,1].
```

All non-target token types should share the same value vector at first.

These value vectors are not general token embeddings. They are the vectors averaged by the attention weights to produce the classifier input.

This design isolates the experiment to the attention score geometry:

- query and key scores can differ by non-target token type
- all non-target values still mean "non-target evidence"
- the attention output remains interpretable as target evidence versus non-target evidence

With shared non-target values, the attention output still has the simple form:

```math
\mathrm{attention\ output}=(p_t,1-p_t).
```

If non-target values were also distinct, then the output would become:

```math
p_tV_t+\sum_{k=1}^{m}p_{u_k}V_{u_k}.
```

That would mix two effects:

- which non-target keys receive attention
- what the attended non-target value vectors mean to the classifier

For the first Stage 3D pass, this extra complication should be avoided. Distinct or learned value vectors can be tested later after the attention denominator effect is understood.

### Positive Inputs

For the first Stage 3D version, keep the target at the beginning to isolate the effect of multiple non-target token types:

```text
t, u_{i_1}, u_{i_2}, ..., u_{i_{n-1}}
```

Each $u_{i_j}$ is sampled from the available non-target token ids.

Positive examples contain exactly one target.

### Negative Inputs

Negative examples contain only non-target tokens:

```text
u_{i_1}, u_{i_2}, ..., u_{i_n}
```

### Non-Target Sampling

Add a CLI option:

```text
--non-target-sampling uniform
```

For the first implementation, only `uniform` is required.

Uniform sampling means every non-target token type has equal probability at each non-target position.

Later variants can add skewed distributions, but they are not required for the first Stage 3D pass.

## Required Implementation Changes

### Step 1: Extend Dataset Generation

Modify Stage 3 dataset generation so it supports multiple non-target token ids.

Required behavior:

- `non_target_token_count=1` reproduces current Stage 3 behavior.
- `non_target_token_count=m` samples non-target positions from token ids `1` through `m`.
- positives still contain exactly one target token.
- negatives contain zero target tokens.

### Step 2: Separate Score Inputs From Attention Values

The model currently assumes two token ids and fixed one-hot values. In that setup, the same one-hot matrix is used both for score computation and as the attention value matrix.

For Stage 3D, update the implementation so the score inputs can distinguish all token types:

```text
0, 1, 2, ..., m
```

Then update fixed attention value construction so:

- token `0` maps to `[1,0]`
- tokens `1` through `m` map to `[0,1]`

The query and key projections should still receive token-specific one-hot inputs or embeddings sufficient to distinguish the non-target token types.

The important constraint is:

**different non-target token ids may have different key scores, but they should initially share the same value meaning.**

### Step 3: Preserve Backward Compatibility

Existing Stage 3, Stage 3B, and Stage 3C commands should still work with default options.

Defaults:

```text
--non-target-token-count 1
--non-target-sampling uniform
```

### Step 4: Add Multi-Non-Target Metrics

Save the existing metrics:

- accuracy
- positive accuracy
- negative accuracy
- mean target attention
- mean delta
- learned alpha coefficient
- classifier weights

Add new metrics:

- number of non-target token types
- mean target score $a$
- per-type non-target score $b_k$
- per-type margin $\Delta_k=a-b_k$
- minimum margin $\Delta_{\min}$
- maximum non-target score
- standard deviation of non-target type scores
- $c\Delta_{\min}$ for learned-log runs
- per-type denominator contribution

Recommended CSV:

```text
non_target_type_metrics.csv
```

Each row should include:

- length
- alpha mode
- non-target token count
- non-target token id
- mean non-target count in sequence
- mean non-target score
- mean margin from target
- mean denominator contribution

### Step 5: Add Theory Prediction For Multi-Non-Target Case

For each positive example, compute:

```math
p_t(n)
=
\frac{1}
{1+\sum_{k=1}^{m} c_k(n)e^{-\alpha\Delta_k}}.
```

Compare this with empirical target attention.

The exact formula should match empirical attention if the recorded per-type counts and scores are correct. The more important question is not whether the algebra matches, but whether the learned score margins satisfy the long-length condition.

### Step 6: Smoke Test

Run a small smoke test with multiple non-target types:

```text
$env:PYTHONPATH = 'src'; ..\.venv\Scripts\python.exe src\stage3_simplified_attention.py --smoke-test --alpha-mode learned_log --non-target-token-count 4 --output-dir runs/stage3d_multiple_non_targets/smoke
```

Expected:

- `model.pt` is created
- `metrics_by_length.csv` is created
- `non_target_type_metrics.csv` is created
- no target token appears in negative examples
- positive examples contain exactly one target
- no NaN appears in key metrics
- `non_target_token_count=1` still reproduces old behavior

## Required Experiment Runs

Run all three attention multiplier modes:

```text
constant
log
learned_log
```

Use the same broad length sweep as Stage 3:

```text
eval_lengths = 10, 100, 1000, 10000, 100000, 1000000, 5000000, 10000000
```

Recommended output root:

```text
runs/stage3d_multiple_non_targets/
```

### Main Conditions

Start with:

```text
non_target_token_count = 4
train_lengths = [10]
test_examples = 50
eval_batch_size = 8
```

Run:

| Run | Alpha mode | Epoch equivalent | Max train steps |
|---|---|---:|---:|
| `constant_e50_nt4` | `constant` | 50 | 1600 |
| `constant_e100_nt4` | `constant` | 100 | 3200 |
| `constant_e1000_nt4` | `constant` | 1000 | 32000 |
| `log_e50_nt4` | `log` | 50 | 1600 |
| `learned_log_e50_nt4` | `learned_log` | 50 | 1600 |
| `learned_log_e100_nt4` | `learned_log` | 100 | 3200 |
| `learned_log_e200_nt4` | `learned_log` | 200 | 6400 |

### Scaling Conditions

If the main conditions work, repeat a smaller subset for:

```text
non_target_token_count = 2
non_target_token_count = 8
```

Minimum recommended subset:

| Run | Alpha mode | Max train steps |
|---|---|---:|
| `constant_e100_nt2` | `constant` | 3200 |
| `log_e50_nt2` | `log` | 1600 |
| `learned_log_e200_nt2` | `learned_log` | 6400 |
| `constant_e100_nt8` | `constant` | 3200 |
| `log_e50_nt8` | `log` | 1600 |
| `learned_log_e200_nt8` | `learned_log` | 6400 |

## Analysis Plan

### 1. Check Whether Non-Target Scores Collapse

Ask:

```text
Are all b_k approximately equal?
```

If yes, the model has effectively recreated the two-score assumption:

```math
S_n\approx(a,b,b,\ldots,b).
```

This would mean Stage 3D did not really break the original mechanism.

### 2. Check Worst-Case Margin

Compute:

```math
\Delta_{\min}=\min_k(a-b_k).
```

For learned-log runs, check:

```math
c\Delta_{\min}.
```

Interpretation:

- if $c\Delta_{\min}>1$, the run satisfies the multi-non-target asymptotic condition
- if $c\Delta_{\min}<1$, the run is expected to eventually fail at sufficiently long length

### 3. Identify Denominator-Dominant Non-Target Types

For each non-target type, compute its denominator contribution:

```math
c_k(n)e^{\alpha b_k}.
```

or after normalization by the target term:

```math
c_k(n)e^{-\alpha\Delta_k}.
```

Ask:

```text
Does one non-target type dominate the denominator?
```

If yes, long-length failure may be controlled by a single hard non-target token type.

### 4. Compare Constant, Log, And Learned-Log

Expected qualitative behavior:

- constant multiplier should still fail at long length unless margins become unrealistically large
- fixed log multiplier should succeed if every $\Delta_k>1$
- learned-log should succeed asymptotically only if $c\Delta_{\min}>1$

### 5. Compare Against Stage 3

Compare Stage 3D with original Stage 3:

- Does adding non-target token types reduce $\Delta_{\min}$?
- Does it make learned-log training slower?
- Does it create a gap between mean margin and worst-case margin?
- Does the model choose to collapse all non-targets into the same key score?

## Expected Outcomes

### Outcome A: Non-Target Scores Collapse

If all non-target scores become nearly identical:

**The model restores the original two-score assumption by learning to treat all non-target tokens the same.**

This would be a strong simplification strategy.

### Outcome B: Scores Differ But Worst-Case Margin Is Large Enough

If non-target scores differ but $\Delta_{\min}$ is still large enough:

**The model generalizes without needing exact two-score collapse.**

This would support the more general theory based on worst-case margin.

### Outcome C: One Non-Target Type Becomes A Bottleneck

If one non-target token type has a much smaller margin:

**The long-length behavior is controlled by the hardest non-target token.**

This would show that mean margin is not enough; the correct diagnostic is $\Delta_{\min}$ and denominator contribution.

### Outcome D: Learned-Log Does Not Reach $c\Delta_{\min}>1$

If learned-log does not cross the threshold:

**The architecture may be capable in theory, but optimization may not push the worst-case margin and learned multiplier into the asymptotic regime.**

This would be similar to the Stage 3B negative result.

## Success Criteria

This task is complete when:

- Stage 3 supports multiple non-target token types
- defaults preserve existing Stage 3 behavior
- constant, log, and learned-log conditions are run
- per-type non-target score metrics are saved
- empirical target attention is compared with the generalized denominator formula
- the report states whether success is controlled by two-score collapse or worst-case margin
- conclusions clearly separate reduced-model behavior from full-transformer behavior

## Notes

- Keep exactly one target token in positive examples.
- Keep binary classification.
- Keep all non-target values equal to $[0,1]$ in the first version.
- Keep the target at position 0 in the first Stage 3D version to isolate the multiple-non-target effect.
- Combining Stage 3C target-anywhere with Stage 3D multiple non-target tokens is a later follow-up.
- Adding skewed non-target sampling is a later follow-up.
