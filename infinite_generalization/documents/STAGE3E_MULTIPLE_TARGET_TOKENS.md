# Stage 3E Multiple Target Token Types

## Objective

Stage 3E tests whether the reduced Stage 3 length-aware attention model can learn a **target class detector**, rather than a detector for one specific target token.

In previous Stage 3 experiments, the positive class was defined by one target token:

```text
positive: sequence contains target token t
negative: sequence contains no target token t
```

Stage 3E changes this to a target set:

```math
\text{target tokens }= {t_1,t_2,\ldots,t_H}
```

A positive example contains exactly one token from the target set. A negative example contains no token from the target set.

The central question is:

**Can the reduced model detect membership in a target class, and is long-length behavior controlled by the target type with the smallest margin?**

## Why Stage 3E

Stage 3D increased non-target-side complexity by adding multiple non-target token types. That experiment showed that long-length behavior is governed by the final-query/non-target-key pair with the smallest margin, not by the average non-target margin.

Stage 3E increases target-side complexity. Instead of asking whether the model can detect one target token, it asks whether the model can detect any token from a target class.

This is closer to realistic existential detection tasks, where the model often needs to detect whether a sequence contains any token from a category.

## Experimental Setup

This first Stage 3E experiment is intentionally controlled:

```text
target_token_count = 3
non_target_token_count = 1
target_position_mode = fixed_start
```

Token ids:

```text
target token ids: 0, 1, 2
non-target token id: 3
```

Positive examples:

```text
t_h, u, u, ..., u
```

where $t_h$ is sampled uniformly from the target token set.

Negative examples:

```text
u, u, u, ..., u
```

The target is fixed at position 0, and there is only one non-target token type. This isolates the new variable: multiple target token types.

As in Stage 3D, the score representation and value representation are separated:

- the query/key score representation distinguishes all token ids
- the attention value representation remains binary

The value mapping is:

```math
t_h \mapsto [1,0],
\qquad
u \mapsto [0,1].
```

Therefore, the attention output remains interpretable as:

```math
(p_{\mathrm{target}},1-p_{\mathrm{target}}).
```

Because each positive example contains exactly one target token, $p_{\mathrm{target}}$ is the attention mass on that target position.

## Theory

In this controlled setup, the final query is always the single non-target query. For a positive example with target type $h$, the final-query attention score row is:

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

The relevant bottleneck is the target type with the smallest margin:

```math
\Delta_{\min}=\min_h\Delta_h.
```

For learned-log attention:

```math
\alpha=1+c\log(1+n),
```

the asymptotic diagnostic is:

```math
c\Delta_{\min}>1.
```

Stage 3E therefore does not change the softmax dilution mechanism. It changes the bottleneck from one target-vs-non-target margin to the target type with the smallest margin.

Later, if multiple non-target types are added, the margin naturally generalizes to:

```math
\Delta_{r,h,k}=a_{r,h}-b_{r,k},
```

where $r$ is the final query token type, $h$ is the target token type, and $k$ is the non-target token type.

## Base Experiment

All base runs used:

```text
target_token_count = 3
non_target_token_count = 1
target_position_mode = fixed_start
train_lengths = [10]
test_examples = 720
eval_chunk_examples = 36
eval_sampling_mode = stratified
eval_batch_size = 8
```

Each evaluation length is generated in chunks of `eval_chunk_examples` to avoid building the full tensor at 10M, and positive examples are stratified so each target token id is generated evenly, without random-draw bias.

Output root:

```text
runs/stage3e/
```

| Run | Multiplier mode | Max train steps |
|---|---|---:|
| `constant_e100_t3_nt1` | `constant` | 3200 |
| `log_e50_t3_nt1` | `log` | 1600 |
| `learned_log_e200_t3_nt1` | `learned_log` | 6400 |

## Base Results At Length 10M

| Run | Positive acc | Negative acc | Positive logit | Target attention | Mean $\Delta_{\min}$ | Worst observed $\Delta_{\min}$ | Mean $c\Delta_{\min}$ | Worst observed $c\Delta_{\min}$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `constant_e100_t3_nt1` | 0.0 | 1.0 | -4.583 | 0.001678 | 9.690 | 9.280 | n/a | n/a |
| `log_e50_t3_nt1` | 1.0 | 1.0 | 3.253 | 1.000000 | 4.294 | 4.033 | n/a | n/a |
| `learned_log_e200_t3_nt1` | 1.0 | 1.0 | 6.431 | 0.999980 | 8.865 | 8.485 | 1.144 | 1.095 |

The qualitative pattern matches earlier Stage 3 results:

- Constant multiplier fits short lengths but fails at long length.
- Fixed log multiplier succeeds because all target-type margins are above 1.
- Learned-log e200 succeeds and crosses the asymptotic diagnostic threshold, but the threshold margin is modest.

## Constant Multiplier Behavior

The constant multiplier run succeeds near the training length but fails between length 10000 and 100000.

| Length | Positive acc | Positive logit | Target attention |
|---:|---:|---:|---:|
| 10 | 1.0 | 4.619 | 0.999418 |
| 10000 | 1.0 | 1.080 | 0.615645 |
| 100000 | 0.0 | -3.283 | 0.142676 |
| 10000000 | 0.0 | -4.583 | 0.001678 |

This is the expected fixed-margin dilution behavior. Even though the learned margins are large at length 10, constant attention scaling does not grow with length, so the non-target denominator eventually dominates.

## Fixed Log Behavior

The fixed-log run succeeds at all evaluated lengths. At length 10M, every target type has positive accuracy 1.0 and target attention 1.0.

| Target token id | Positive acc at 10M | Target score | Target attention | Margin |
|---:|---:|---:|---:|---:|
| 0 | 1.0 | 2.451 | 1.000000 | 4.456 |
| 1 | 1.0 | 2.028 | 1.000000 | 4.033 |
| 2 | 1.0 | 2.388 | 1.000000 | 4.393 |

The smallest observed margin is approximately:

```math
\Delta_{\min}\approx4.033>1.
```

This is comfortably above the fixed-log threshold.

## Learned-Log Behavior

The learned-log e200 run also succeeds at all evaluated lengths.

At length 10M:

| Target token id | Positive acc | Target score | Target attention | Margin | $c\Delta$ |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.0 | 4.858 | 0.999993 | 9.098 | 1.174 |
| 1 | 1.0 | 4.245 | 0.999955 | 8.485 | 1.095 |
| 2 | 1.0 | 4.773 | 0.999991 | 9.013 | 1.163 |

The learned coefficient is:

```math
c\approx0.129.
```

The weakest target type is target token id `1`, with:

```math
c\Delta_{\min}\approx1.095.
```

This crosses the asymptotic diagnostic threshold, but only with a modest margin. This is important because it shows that multiple target types make the learned-log solution more constrained: the model must push every target type above the threshold, not just the average target score.

## Target-Type Bottleneck Analysis

Target token id `1` is the smallest-margin target type in all three base runs.

| Run | Smallest-margin target id | Worst observed margin |
|---|---:|---:|
| `constant_e100_t3_nt1` | 1 | 9.280 |
| `log_e50_t3_nt1` | 1 | 4.033 |
| `learned_log_e200_t3_nt1` | 1 | 8.485 |

This supports the main Stage 3E diagnostic:

**The target class is only as robust as its smallest-margin target type.**

Average positive accuracy is not enough. A model can look strong on aggregate while one target type is closer to the failure boundary.

## Interpretation

Stage 3E does not change the underlying length-generalization mechanism.

The same softmax denominator effect remains:

```math
p_t(n\mid h)
=
\frac{1}
{1+(n-1)e^{-\alpha\Delta_h}}.
```

What changes is the bottleneck. In the single-target setup, there is only one target-vs-non-target margin. In Stage 3E, each target type has its own margin:

```math
\Delta_h=a_h-b.
```

Therefore, long-length behavior is controlled by:

```math
\min_h\Delta_h.
```

The base result shows that the reduced model can learn a target class detector with three target token types. However, the learned-log solution is tighter than in simpler settings because the weakest target type has worst observed $c\Delta_{\min}\approx1.095$.

## Follow-Up Experiments

### Increase Target Token Count

The most direct next follow-up is to increase the number of target token types:

```text
target_token_count = 2, 4, 8
non_target_token_count = 1
target_position_mode = fixed_start
```

Main question:

```text
Does worst observed cDelta_min drop below 1 as the number of target types increases?
```

This is the natural next test because learned-log e200 crossed the threshold only modestly in the base run.

### Add Multiple Non-Target Types

After target-count scaling, combine Stage 3E with Stage 3D:

```text
target_token_count = 3
non_target_token_count = 4
target_position_mode = fixed_start
```

The bottleneck should become:

```math
\Delta_{\min}
=
\min_{h,k}(a_h-b_k).
```

This tests whether target-side and non-target-side bottlenecks combine cleanly.

### Combine With Target-Anywhere Placement

Then combine Stage 3E with Stage 3C:

```text
target_token_count = 3
target_position_mode = nonfinal_random
```

The goal is to confirm that target position remains unimportant when target types are multiple.

### Full Stage 3C+D+E Combination

The final sanity check would combine all three extensions:

```text
multiple target token types
multiple non-target token types
target anywhere except final position
```

This should be treated as a confirmation run, not as the next primary research direction, unless earlier follow-ups reveal unexpected behavior.

## Current Conclusion

**Stage 3E succeeds in the controlled base setup.**

The reduced model learns to detect a target class with three target token types. Constant attention scaling still fails at long length, fixed-log succeeds, and learned-log e200 succeeds by pushing even the weakest target type above the asymptotic diagnostic threshold.

The main new finding is:

**Long-length behavior is controlled by the target token type with the smallest margin, not by the average target score.**

## Limitations

- Only `target_token_count=3` was tested.
- Only `non_target_token_count=1` was tested.
- The target was fixed at position 0.
- Only one seed was analyzed.
- Positive examples contained exactly one target token.
- Multiple target occurrences were not tested.
- This result applies only to the reduced no-positional-encoding model.
