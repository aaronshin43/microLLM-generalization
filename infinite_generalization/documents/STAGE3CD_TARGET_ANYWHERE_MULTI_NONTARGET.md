# Stage 3C+D Target Anywhere With Multiple Non-Target Tokens

## Objective

Stage 3C+D is a short sanity check that combines two previous Stage 3 extensions:

- Stage 3C: the target token can appear at any non-final position.
- Stage 3D: the sequence can contain multiple non-target token types.

The purpose of this check is narrow:

**Does allowing the target to appear anywhere change the Stage 3D multiple-non-target conclusion?**

The expected answer is no. The reduced model has no positional encoding, so target position should not materially affect the attention score assigned to the target token. The main difficulty should still be the multiple-non-target denominator and the smallest target-vs-non-target margin.

## Setup

Positive examples contain exactly one target token sampled from any non-final position:

```math
p\in\{0,1,\ldots,n-2\}.
```

The final token remains non-target, so the final readout query is still a non-target query. This keeps the Stage 3D final-query-conditioned analysis applicable.

Negative examples contain only non-target tokens.

All runs used:

```text
non_target_token_count = 4
target_position_mode = nonfinal_random
train_lengths = [10]
test_examples = 50
eval_batch_size = 8
```

Output root:

```text
runs/stage3cd_target_anywhere_multi_nontarget/
```

## Runs

| Run | Multiplier mode | Max train steps |
|---|---|---:|
| `constant_e100_nt4_target_anywhere` | `constant` | 3200 |
| `log_e50_nt4_target_anywhere` | `log` | 1600 |
| `learned_log_e200_nt4_target_anywhere` | `learned_log` | 6400 |

These are representative runs rather than a full grid. The goal is to check whether target-anywhere placement changes the Stage 3D result.

## Results At Length 10M

| Run | Positive acc | Positive logit | Target attention | Mean $\Delta_{\min}$ | Worst observed $\Delta_{\min}$ | Mean $c\Delta_{\min}$ | Worst observed $c\Delta_{\min}$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `constant_e100_nt4_target_anywhere` | 0.0 | -4.521 | 0.001845 | 9.628 | 9.329 | n/a | n/a |
| `log_e50_nt4_target_anywhere` | 1.0 | 3.212 | 1.000000 | 4.216 | 4.048 | n/a | n/a |
| `learned_log_e200_nt4_target_anywhere` | 1.0 | 6.411 | 0.999998 | 8.510 | 8.256 | 1.292 | 1.253 |

The qualitative behavior matches Stage 3D:

- The constant multiplier run still fails at long length.
- The fixed-log run still succeeds.
- The learned-log e200 run still succeeds and remains above the asymptotic diagnostic threshold, with worst observed $c\Delta_{\min}\approx1.253>1$.

## Comparison To Fixed-Start Stage 3D

The target-anywhere runs are very close to the original fixed-start Stage 3D runs.

| Condition | Fixed-start positive acc at 10M | Target-anywhere positive acc at 10M | Fixed-start positive logit | Target-anywhere positive logit |
|---|---:|---:|---:|---:|
| `constant_e100_nt4` | 0.0 | 0.0 | -4.515 | -4.521 |
| `log_e50_nt4` | 1.0 | 1.0 | 3.232 | 3.212 |
| `learned_log_e200_nt4` | 1.0 | 1.0 | 6.412 | 6.411 |

This supports the interpretation that fixing the target at position 0 was not driving the Stage 3D result.

## Target-Position Bucket Check

At length 10M, the target-position buckets followed the run-level outcome.

| Run | Bucket | Positive acc | Target attention | Mean delta |
|---|---|---:|---:|---:|
| `constant_e100_nt4_target_anywhere` | `beginning` | 0.0 | 0.001942 | 9.883 |
| `constant_e100_nt4_target_anywhere` | `middle` | 0.0 | 0.001738 | 9.763 |
| `constant_e100_nt4_target_anywhere` | `end_nonfinal` | 0.0 | 0.001899 | 9.812 |
| `log_e50_nt4_target_anywhere` | `beginning` | 1.0 | 1.000000 | 4.375 |
| `log_e50_nt4_target_anywhere` | `middle` | 1.0 | 1.000000 | 4.312 |
| `log_e50_nt4_target_anywhere` | `end_nonfinal` | 1.0 | 1.000000 | 4.338 |
| `learned_log_e200_nt4_target_anywhere` | `beginning` | 1.0 | 0.999999 | 8.743 |
| `learned_log_e200_nt4_target_anywhere` | `middle` | 1.0 | 0.999998 | 8.638 |
| `learned_log_e200_nt4_target_anywhere` | `end_nonfinal` | 1.0 | 0.999998 | 8.684 |

There is no evidence that one target-position bucket fails earlier. The constant run fails across all buckets, while the fixed-log and learned-log e200 runs succeed across all buckets.

## Denominator-Dominant Non-Target Type

In this target-anywhere sanity check, the denominator-dominant non-target type changed from token id `3` in the fixed-start Stage 3D main runs to token id `2`.

At length 10M:

| Run | Dominant non-target type | Mean margin | Denominator fraction |
|---|---:|---:|---:|
| `constant_e100_nt4_target_anywhere` | 2 | 9.638 | 0.291 |
| `log_e50_nt4_target_anywhere` | 2 | 4.232 | 0.525 |
| `learned_log_e200_nt4_target_anywhere` | 2 | 8.523 | 0.364 |

This does not change the main interpretation. The identity of the denominator-dominant non-target type can vary by run, seed, or final-query composition. The important diagnostic is still the worst-case margin and denominator contribution, not the specific token id.

## Conclusion

**Stage 3C+D confirms that the Stage 3D result is not an artifact of placing the target at position 0.**

When the target can appear at any non-final position, the reduced model behaves almost the same as in the fixed-start Stage 3D setup. Target position does not materially change positive accuracy, target attention, or the learned-log threshold behavior.

The main Stage 3D conclusion remains unchanged:

**With multiple non-target token types, long-length behavior is governed by the final-query/non-target-key pair with the smallest margin, not by target position.**

## Limitations

- This was a small sanity check, not a full experimental grid.
- Only `non_target_token_count=4` was tested.
- The run uses one seed.
- The target is still excluded from the final position.
- This result applies only to the reduced no-positional-encoding model.
