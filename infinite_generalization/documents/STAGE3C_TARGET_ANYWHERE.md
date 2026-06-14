# Stage 3C Target Can Appear Anywhere

## Objective

Stage 3C tests whether the reduced Stage 3 model still works when the target token can appear at any non-final position.

In the original Stage 3 setup, every positive example had the fixed form:

```text
t, u, u, ..., u
```

This placed the target token at position 0. Stage 3C removes that fixed-position simplification while preserving the key theoretical condition:

```math
q_{\mathrm{last}}=q_u.
```

That condition is preserved by forcing the final token to remain the non-target token $u$. Therefore, in a length-$n$ positive example, the target position is sampled from:

```math
p \in \{0,1,\ldots,n-2\}.
```

The central question is:

**Does the reduced model learn a position-independent target detector when the target can appear anywhere except the final readout position?**

## Setup

The model is unchanged from Stage 3:

- two tokens: target $t=0$ and non-target $u=1$
- fixed one-hot values: $t\mapsto[1,0]$ and $u\mapsto[0,1]$
- learned query projection matrix
- learned key projection matrix
- values equal to the one-hot inputs
- last-query attention
- linear classifier on the attention output
- no positional encoding

Positive examples contain exactly one target token:

```text
u, ..., t, ..., u
```

with the target position sampled uniformly from the non-final positions.

Negative examples remain:

```text
u, u, u, ..., u
```

The final position is excluded because if the target appeared at the final position, then the final query would become the target query rather than the non-target query:

```math
q_{\mathrm{last}}=q_t.
```

That would be a different theoretical setup. Stage 3C only tests whether target position matters when the readout query remains fixed as the non-target query.

## Expected Theory

Because the model has no positional encoding, the key score for a token should depend only on token identity, not on position.

For a positive example with the target at position $p$, the score row should have the form:

```math
S_n^{(p)}
=
(b,\ldots,b,a,b,\ldots,b),
```

where:

- $a$ is the target key score.
- $b$ is the shared non-target key score.
- the location of $a$ changes with target position $p$.
- every non-target key still receives the same score $b$.

The two-score assumption therefore becomes position-independent:

```math
\Delta=a-b.
```

The target attention mass is still:

```math
p_t(n)
=
\frac{e^{\alpha\Delta}}
{e^{\alpha\Delta}+(n-1)}.
```

The formula does not depend on the target position $p$. It depends only on the fact that there is one target key and $n-1$ non-target keys.

## Implementation

Stage 3C added:

- `--target-position-mode fixed_start`
- `--target-position-mode nonfinal_random`

The default remains:

```text
fixed_start
```

for backward compatibility with the original Stage 3 runs.

In `nonfinal_random` mode:

1. positive examples start as all non-target tokens
2. one target position is sampled from $0$ through $n-2$
3. the target token is placed at that position
4. the final token remains non-target

The evaluation code now uses the actual target position when computing:

- target key score
- non-target key mean score
- target-vs-non-target score margin
- empirical target attention

Stage 3C also writes:

```text
target_position_metrics.csv
```

This file aggregates positive examples by target-position bucket. A target-position bucket is a coarse group of examples based on where the target appears in the sequence, rather than the exact target index. This makes it easier to compare whether early, middle, and late non-final targets behave differently.

- `beginning`
- `middle`
- `end_nonfinal`

Each bucket records:

- positive accuracy
- mean target attention
- mean delta
- mean non-target score standard deviation
- target position range
- final target count

## Runs

All Stage 3C runs used:

- `train_lengths = [10]`
- `target_position_mode = nonfinal_random`
- `test_examples = 50`
- `eval_batch_size = 8`
- evaluation lengths up to 10M

Run output root:

```text
runs/stage3c_target_anywhere/
```

The seven Stage 3 conditions were rerun:

| Run | Alpha mode | Max train steps |
|---|---|---:|
| `constant_e50` | `constant` | 1600 |
| `constant_e100` | `constant` | 3200 |
| `constant_e1000` | `constant` | 32000 |
| `log_e50` | `log` | 1600 |
| `learned_log_e50` | `learned_log` | 1600 |
| `learned_log_e100` | `learned_log` | 3200 |
| `learned_log_e200` | `learned_log` | 6400 |

## Overall Results

The 10M-length results were nearly identical to the original fixed-start Stage 3 behavior.

| Run | Mode | Updates | $c$ | $\Delta$ | $c\Delta$ | Positive logit at 10M | Positive accuracy at 10M | Non-target score std |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `constant_e50` | `constant` | 1600 | n/a | 8.0941 | n/a | -3.6417 | 0.0000 | 0.0 |
| `constant_e100` | `constant` | 3200 | n/a | 8.9856 | n/a | -4.9126 | 0.0000 | 0.0 |
| `constant_e1000` | `constant` | 32000 | n/a | 12.2747 | n/a | -18.6486 | 0.0000 | 0.0 |
| `log_e50` | `log` | 1600 | n/a | 3.9333 | n/a | 3.7183 | 1.0000 | 0.0 |
| `learned_log_e50` | `learned_log` | 1600 | 0.0662 | 7.4015 | 0.4902 | -1.3553 | 0.0000 | 0.0 |
| `learned_log_e100` | `learned_log` | 3200 | 0.0966 | 7.9243 | 0.7659 | 4.8679 | 1.0000 | 0.0 |
| `learned_log_e200` | `learned_log` | 6400 | 0.1361 | 8.2918 | 1.1284 | 6.8163 | 1.0000 | 0.0 |

Interpretation:

- Constant multiplier still fails at long length.
- Fixed log multiplier still succeeds because $\Delta>1$.
- Learned-log e50 still fails because $c\Delta<1$.
- Learned-log e100 succeeds through 10M but still has $c\Delta<1$.
- Learned-log e200 still crosses $c\Delta>1$ and remains the asymptotic simplified-model solution.

## Target-Position Bucket Stability

The key Stage 3C diagnostic is whether different target-position buckets behave differently.

Across all evaluation lengths, the bucket-level variation was essentially zero up to floating-point noise:

| Run | Max delta bucket range | Max target-attention bucket range | Min positive accuracy across buckets | Max non-target score std | Final target count |
|---|---:|---:|---:|---:|---:|
| `constant_e50` | $9.5\times10^{-7}$ | $4.8\times10^{-7}$ | 0.0000 | 0.0 | 0 |
| `constant_e100` | $9.5\times10^{-7}$ | $3.6\times10^{-7}$ | 0.0000 | 0.0 | 0 |
| `constant_e1000` | $1.9\times10^{-6}$ | $6.0\times10^{-7}$ | 0.0000 | 0.0 | 0 |
| `log_e50` | $7.2\times10^{-7}$ | $1.8\times10^{-7}$ | 1.0000 | 0.0 | 0 |
| `learned_log_e50` | $9.5\times10^{-7}$ | $2.1\times10^{-5}$ | 0.0000 | 0.0 | 0 |
| `learned_log_e100` | $9.5\times10^{-7}$ | $1.1\times10^{-5}$ | 1.0000 | 0.0 | 0 |
| `learned_log_e200` | $9.5\times10^{-7}$ | $1.8\times10^{-7}$ | 1.0000 | 0.0 | 0 |

Interpretation:

The target-position buckets do not produce meaningfully different behavior. The small nonzero ranges are numerical-scale effects. The final target count is always 0, confirming that positives never placed the target at the final readout position.

## Learned-Log e200 Bucket Example

At length 10M, the strongest learned-log run behaved identically across target-position buckets:

| Bucket | Positive examples | Target position range | Positive accuracy | $\Delta$ | Target attention | Non-target score std |
|---|---:|---|---:|---:|---:|---:|
| `beginning` | 10 | 156203-3209198 | 1.0000 | 8.2918 | 0.999968 | 0.0 |
| `middle` | 7 | 4043097-6469110 | 1.0000 | 8.2918 | 0.999968 | 0.0 |
| `end_nonfinal` | 8 | 7184538-9811564 | 1.0000 | 8.2918 | 0.999968 | 0.0 |

This directly supports the position-independence interpretation.

## Answers To Main Questions

### 1. Does The Model Still Satisfy The Two-Score Assumption?

Yes.

The non-target score standard deviation remained 0.0 in every run and every target-position bucket. This means the model still assigns one score $a$ to the target key and one shared score $b$ to all non-target keys.

### 2. Does Delta Remain Stable Across Target Positions?

Yes.

The maximum bucket-level delta range across all runs and lengths was about $1.9\times10^{-6}$, which is numerical noise. Delta does not meaningfully depend on where the target appears.

### 3. Does Target Attention Remain Stable Across Target Positions?

Yes.

Target attention was stable across beginning, middle, and end-nonfinal buckets. The largest bucket-level attention range was about $2.1\times10^{-5}$, from the undertrained learned-log e50 run. The successful runs were even more stable.

### 4. Does Learned-Log e200 Still Reach The Asymptotic Regime?

Yes.

For learned-log e200:

```math
c\Delta \approx 1.1284 > 1.
```

This remains above the simplified asymptotic threshold.

### 5. Does Any Target-Position Bucket Fail Earlier?

No.

Within each run, bucket-level positive accuracy followed the run-level outcome. Failed runs failed across buckets; successful runs succeeded across buckets. There was no evidence that final-adjacent non-final targets behave differently.

### 6. Does Target-Anywhere Training Change c, Delta, Or Classifier Calibration?

Not meaningfully.

The learned-log values were very close to the original fixed-start Stage 3 reruns:

- learned-log e50: $c\Delta\approx0.4902$
- learned-log e100: $c\Delta\approx0.7659$
- learned-log e200: $c\Delta\approx1.1284$

The qualitative conclusions are unchanged.

## Main Conclusion

**Stage 3C shows that the reduced model learns a position-independent token detector.**

The target can appear anywhere except the final readout position, and the model still behaves according to the same two-score assumption:

```math
S_n^{(p)}=(b,\ldots,b,a,b,\ldots,b).
```

The target position changes where the $a$ appears in the score row, but it does not change the values of $a$, $b$, $\Delta$, target attention, or classification behavior.

This result is expected for the reduced model because it has no positional encoding. It is still useful because it confirms that the original fixed-start Stage 3 result was not an artifact of always placing the target at position 0.

## Limitations

- The target was still excluded from the final position.
- The final readout query was always the non-target query.
- The model still used fixed one-hot values.
- The model still had no positional encoding.
- This result applies only to the reduced model, not automatically to full transformers.

Allowing the target at the final position is a separate follow-up because it changes the readout query from the non-target query to the target query.
