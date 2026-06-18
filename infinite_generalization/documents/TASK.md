# Task: Stage 3C+D Sanity Check

## Objective

Stage 3C+D is a short sanity check that combines the two already completed Stage 3 extensions:

- Stage 3C: the target token can appear at any non-final position.
- Stage 3D: the sequence can contain multiple non-target token types.

The goal is not to open a new major research direction. The goal is to confirm that the Stage 3D result is not an artifact of always placing the target at position 0.

The expected outcome is simple:

**Adding target-anywhere placement to the multiple-non-target setup should not substantially change the Stage 3D conclusions.**

This expectation follows from the reduced model design. The model has no positional encoding, so target position should not affect the attention score assigned to the target token. The main remaining difficulty should still be the multiple-non-target denominator and the worst-case target-vs-non-target margin.

## Background

Stage 3C showed that target position did not meaningfully affect behavior when there was only one non-target token type.

Stage 3D showed that the reduced model can handle multiple non-target token types without collapsing all non-target attention scores into one shared score. The relevant generalized bottleneck was:

```math
\Delta_{\min}=\min_{r,k}(a_r-b_{r,k}),
```

where:

- $r$ is the final readout non-target token type.
- $a_r$ is the target key score under final query type $r$.
- $b_{r,k}$ is the score for non-target token type $k$ under final query type $r$.

For learned-log attention,

```math
\alpha=1+c\log(1+n),
```

the asymptotic diagnostic is:

```math
c\Delta_{\min}>1.
```

Stage 3C+D keeps this same theory, but removes the fixed target-at-position-0 simplification.

## Scope

This is a sanity check, not a full new stage.

Use the existing implementation:

```text
infinite_generalization/src/stage3_simplified_attention.py
```

Do not add new architecture components.

Do not add positional encoding.

Do not change the binary target-present classification task.

Do not run a large grid unless the sanity check produces surprising results.

## Experimental Design

Use:

```text
target_position_mode = nonfinal_random
non_target_sampling = uniform
```

Positive examples:

```text
u_{i_0}, ..., t, ..., u_{i_{n-1}}
```

with exactly one target token sampled from non-final positions:

```math
p\in\{0,1,\ldots,n-2\}.
```

The final token remains non-target so the readout query is still a non-target query.

Negative examples:

```text
u_{i_0},u_{i_1},\ldots,u_{i_{n-1}}.
```

The non-target token ids are sampled uniformly from:

```text
1, 2, ..., m
```

where $m$ is `non_target_token_count`.

## Main Run Set

Use `non_target_token_count=4` as the main sanity-check condition because it matches the Stage 3D main experiment.

Run only the representative conditions:

| Run | Alpha mode | Max train steps | Purpose |
|---|---|---:|---|
| `constant_e100_nt4_target_anywhere` | `constant` | 3200 | should still fail at long length |
| `log_e50_nt4_target_anywhere` | `log` | 1600 | should still succeed if worst-case margins remain above 1 |
| `learned_log_e200_nt4_target_anywhere` | `learned_log` | 6400 | should still satisfy $c\Delta_{\min}>1$ |

Recommended output root:

```text
runs/stage3cd_target_anywhere_multi_nontarget/
```

## Commands

Run from:

```text
infinite_generalization/
```

### Constant Multiplier

```powershell
$env:PYTHONPATH = 'src'; ..\.venv\Scripts\python.exe src\stage3_simplified_attention.py --alpha-mode constant --target-position-mode nonfinal_random --non-target-token-count 4 --train-lengths 10 --test-examples 50 --eval-batch-size 8 --max-train-steps 3200 --output-dir runs/stage3cd_target_anywhere_multi_nontarget/constant_e100_nt4_target_anywhere
```

### Fixed Log Multiplier

```powershell
$env:PYTHONPATH = 'src'; ..\.venv\Scripts\python.exe src\stage3_simplified_attention.py --alpha-mode log --target-position-mode nonfinal_random --non-target-token-count 4 --train-lengths 10 --test-examples 50 --eval-batch-size 8 --max-train-steps 1600 --output-dir runs/stage3cd_target_anywhere_multi_nontarget/log_e50_nt4_target_anywhere
```

### Learned-Log Multiplier

```powershell
$env:PYTHONPATH = 'src'; ..\.venv\Scripts\python.exe src\stage3_simplified_attention.py --alpha-mode learned_log --target-position-mode nonfinal_random --non-target-token-count 4 --train-lengths 10 --test-examples 50 --eval-batch-size 8 --max-train-steps 6400 --output-dir runs/stage3cd_target_anywhere_multi_nontarget/learned_log_e200_nt4_target_anywhere
```

## Metrics To Inspect

Use the existing CSV outputs:

```text
metrics_by_length.csv
non_target_type_metrics.csv
target_position_metrics.csv
```

Main metrics:

- positive accuracy by length
- mean positive logit by length
- empirical target attention by length
- mean $\Delta_{\min}$
- worst observed $\Delta_{\min}$
- mean and worst observed $c\Delta_{\min}$ for learned-log
- non-target type-score standard deviation
- denominator-dominant non-target token type
- target-position bucket accuracy and margin

## Key Questions

### 1. Does Target Position Change The Stage 3D Outcome?

Expected answer:

No. The results should be close to the fixed-start Stage 3D runs.

### 2. Does Any Target-Position Bucket Fail Earlier?

Check:

```text
target_position_metrics.csv
```

Compare:

- `beginning`
- `middle`
- `end_nonfinal`

Expected answer:

No bucket should behave meaningfully differently, except for numerical noise and finite-sample variation.

### 3. Does The Multiple Non-Target Bottleneck Remain The Main Difficulty?

Check:

```text
non_target_type_metrics.csv
```

Expected answer:

Yes. The relevant bottleneck should still be the hardest final-query/non-target-key margin, not the target position.

### 4. Does Learned-Log Still Cross The Threshold?

For `learned_log_e200_nt4_target_anywhere`, check:

```math
c\Delta_{\min}>1.
```

Expected answer:

Yes, if the run behaves like the Stage 3D fixed-start result.

## Interpretation Rules

If results match Stage 3D:

**Treat Stage 3C+D as a control result.** It confirms that fixing the target at position 0 was not driving the Stage 3D conclusions.

If target-position buckets differ:

Inspect whether the difference is caused by finite sample noise, target position, or interaction with final readout query type.

If learned-log no longer crosses the threshold:

Rerun with a different seed before drawing conclusions. This task is small enough that one seed should not be overinterpreted.

## Reporting Plan

Do not create a large standalone report unless the result is surprising.

Preferred reporting:

- Add a short Stage 3C+D subsection to `STAGE3D_MULTIPLE_NON_TARGET_TOKENS.md`, or
- Add a short note to the main Stage 3 report.

The report should state:

- this was a sanity check
- the target was sampled from non-final positions
- multiple non-target token types were still present
- whether the results matched Stage 3D
- whether target-position buckets behaved similarly

## Success Criteria

This task is complete when:

- the three sanity-check runs finish
- required CSV files are produced
- target-position bucket metrics are inspected
- the result is summarized briefly in the documentation

## Expected Conclusion

The expected conclusion is:

**Stage 3C+D does not materially change the Stage 3D result. Target position remains unimportant in the reduced no-positional-encoding model, while the multiple-non-target worst-case margin remains the meaningful diagnostic.**
