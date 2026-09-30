# Learned-Log Parameter-Selection Follow-up Results

All three phases in the [launch specification](MECHANISM_FOLLOWUP_SPEC.md) finished:
15 new-seed runs, 20 new coefficient-initialization runs, and 18 fresh update
traces. A separate one-run trace probe passed before the full trace. New training
took 1337.9 seconds (22.3 minutes) including that probe, below the provisional
60-minute ceiling. All runs used 6400 updates; no checkpoint was loaded to
resume AdamW.

## Main findings

1. **The exponent ordering replicated across independent initializations.**
   At 6400 updates, each new seed 3–7 had `[10] > [10,100] > [100]` in
   $c\Delta$. The mixture-minus-`[100]` difference was positive in all five,
   mean 0.0356 ± 0.0202 sample SD, range 0.0139–0.0654. Combined with the
   earlier seeds 0–2, the sign is positive in 8/8 paired seeds. This supports
   the observed ordering for this fixed setup, while the small mixture gap and
   seed spread still matter.
2. **The threshold pattern also replicated at the original initialization.**
   All five new `[10]` runs first appeared above strict $c\Delta=1$ at a
   200-update recording between 4200 and 5400; none of the ten `[100]` or
   mixture runs crossed by 6400. No run that crossed was later observed below
   1 through 6400. The precise crossing lies within the preceding 200-update
   bracket. This is not an indefinite persistence claim.
3. **Changing initial $c$ changed more than the speed of threshold entry.**
   At $c_0=0.1$, all ten single-length runs fitted their classes and crossed
   $c\Delta=1$ by 6400. At $c_0=1.0$, four paired seeds fitted and crossed
   early, but seed 5 failed to classify positives for *both* training lengths.
   Its final head slope and raw $\Delta$ stayed negative. The `[10]`–`[100]`
   raw-margin ordering also reversed between the original and larger initial
   coefficients among fitted runs. This is substantial initialization
   sensitivity, even though the exponent ordering survived among fitted runs.
4. **The update trace narrows the learning-path explanation.** With the original
   small coefficient, the query/key contribution to the first 200 updates'
   $c\Delta$ change had larger absolute magnitude than the coefficient
   contribution in all six traced seed/length cells. Signs differed across
   seeds, so there was no universal early *increase* in raw margin. In the
   windows 1590–1610 and 3190–3210, coefficient changes became the larger
   contribution, and the `[10]` coefficient contribution exceeded `[100]` in
   each traced seed 3–5. At $c_0=1.0$, seed 5 instead drove its already
   negative margin further down and never reached the fitted branch by 6400.

## Independent-seed replication

Final $c\Delta$ by paired new seed:

| Seed | `[10]` | `[100]` | `[10,100]` | Mixture minus `[100]` | First `[10]` recording above 1 |
|---:|---:|---:|---:|---:|---:|
| 3 | 1.1003 | 0.6959 | 0.7098 | +0.0139 | 5400 |
| 4 | 1.2080 | 0.7614 | 0.7812 | +0.0198 | 4200 |
| 5 | 1.3175 | 0.5918 | 0.6311 | +0.0393 | 4400 |
| 6 | 1.2047 | 0.7993 | 0.8386 | +0.0393 | 4200 |
| 7 | 1.2000 | 0.7991 | 0.8645 | +0.0654 | 4200 |

The mixture still had a *smaller* raw $\Delta$ than `[100]` in all five new
seeds (paired differences -0.495 to -0.539, except seed 7 at -0.462).
The head-aware analytical failure-length rank was less stable than the exponent
rank: at 6400, four of five new seeds ranked `[10] > [10,100] > [100]`, while
seed 3 ranked `[10] > [100] > [10,100]`. The 1600 rankings varied further;
seed 5 was not comparable under the fitting/head checks then. Thus the original
three-seed reversal of the *entire* failure-length ranking does not replicate
uniformly. These are analytical predictions, not measured long-sequence
failures. [Per-run values](../../runs/training_length/seed_replication_20260929_01/analysis/replication_per_run.csv)
and [paired ranks](../../runs/training_length/seed_replication_20260929_01/analysis/replication_paired.csv)
retain all eight seeds.

## Initial-coefficient intervention

The intervention changed only the learned scalar's initial value within each
paired seed; initial query, key and head states were bitwise identical across
all six single-length/initialization cells. The actual initial $c$ values were
0.006715, 0.100000 and 1.000000. Consequently, initial attention differed.
Every run still processed 409600 balanced sequences. A length-10 run processed
4.096 million tokens, and a length-100 run 40.96 million.

| Initial $c$ | Fitted pairs | Mean final $c\Delta$, `[10]` | Mean final $c\Delta$, `[100]` | Mean raw $\Delta$ difference, `[10]` minus `[100]` |
|---:|---:|---:|---:|---:|
| 0.006715 | 5/5 | 1.2061 | 0.7295 | -1.4906 |
| 0.1 | 5/5 | 2.5317 | 1.8241 | +0.3482 |
| 1.0 | 4/5 | 4.3515 (fitted four) | 2.9826 (fitted four) | +0.5987 (fitted four) |

The excluded seed-5 pair at $c_0=1.0$ ended at $c\Delta=-4.2072$ (`[10]`)
and -2.8201 (`[100]`); both had 0.0 positive accuracy and 1.0 negative
accuracy in the explicit length-10000 evaluation. It must not be counted as
a successful reversal of learned exponent selection. At $c_0=0.1$, seed 5
fitted both classes, but `[100]` was still much less optimized on common
length-100 BCE at 6400 than its paired `[10]` run; comparisons that assume
equal progress need to exclude that unmatched case.

First 200-update recordings above $c\Delta=1$ for the two single lengths;
`>6400` means right-censored:

| Seed | Baseline `[10]` / `[100]` | $c_0=0.1$ `[10]` / `[100]` | $c_0=1.0$ `[10]` / `[100]` |
|---:|---:|---:|---:|
| 3 | 5400 / >6400 | 400 / 400 | 400 / 400 |
| 4 | 4200 / >6400 | 400 / 400 | 400 / 600 |
| 5 | 4400 / >6400 | 3000 / 6400 | >6400 / >6400 (failed fit) |
| 6 | 4200 / >6400 | 400 / 400 | 200 / 200 |
| 7 | 4200 / >6400 | 600 / 400 | 400 / 400 |

No post-crossing observation fell back below 1. A crossing first observed
at 6400 has only that single above-1 observation. Matching a common length-100
balanced BCE by interpolation at 0.04, 0.02, 0.01 and 0.005 left a positive
`[10]` minus `[100]` exponent gap in every available matched pair: 5/5 at
baseline and 4/5 at each larger initialization. Seed 5 at $c_0=0.1$ lacked
the shared low-loss range; seed 5 at $c_0=1.0$ did not fit. This matching is
descriptive, not causal or a convergence guarantee. Full [initial conditions](../../runs/training_length/coefficient_init_20260929_01/initial_states.csv),
[intervention runs](../../runs/training_length/seed_replication_20260929_01/analysis/intervention_per_run.csv),
and [loss matches](../../runs/training_length/seed_replication_20260929_01/analysis/intervention_loss_matched.csv)
are available.

![New-seed trajectories and initial-coefficient response](../../runs/training_length/seed_replication_20260929_01/analysis/followup_overview.png)

## What the realized updates show

The trace sampled every update 1–200 and 21-update windows around 1600 and
3200, plus the neighborhood of a sampled growth-threshold crossing when one
occurred. For every selected step it stored the unconstrained coefficient's
gradient and applied update, query/key gradients and relevant parameter
updates, raw $\Delta$, head state, positive target mass at lengths 10/100,
and the exact two-score logits before/after. All 18 full replays matched their
source runs bitwise in final weights and exactly in the standard diagnostics
and history. The largest decomposition residual across the full traces was
$6.8\times10^{-15}$; the one-run probe's largest residual was
$2.4\times10^{-15}$.

For a step, $\delta(c\Delta)$ was split into $\bar\Delta\,\delta c$ and
$\bar c\,\delta\Delta$, and the raw-gap change was split into midpoint query
and key contributions. The same accounting was checked for effective margin
at lengths 10 and 100. This uses applied AdamW changes rather than inferring
changes from raw gradients. For example, seed 3's first baseline step had
unconstrained-coefficient gradients about -2.29e-5 (`[10]`) and -4.55e-6
(`[100]`), a fivefold difference, but its applied parameter updates were
0.002999 and 0.002994. Raw gradient size alone would misstate that step's
effect.

At baseline initialization, the first 200-step cumulative score contribution
to $c\Delta$ exceeded the coefficient contribution in absolute value in all
six traced cells. Some early changes were negative as the initially negative
head slope and raw $\Delta$ moved through a transient. At updates 1590–1610,
the following mean signed contributions across traced seeds 3–5 show the
later shift:

| Training length | From $c$ update | From raw $\Delta$ update | Mean query contribution to raw $\Delta$ | Mean key contribution to raw $\Delta$ |
|---|---:|---:|---:|---:|
| `[10]` | +0.00540 | +0.00143 | +0.01481 | +0.01527 |
| `[100]` | +0.00265 | +0.00108 | +0.01520 | +0.01563 |

These are 21-update **sums per run, averaged across three seeds**, not
per-step gradients. In each of the three seeds, the coefficient contribution
was greater for `[10]` in both the 1590–1610 and 3190–3210 windows. Query
and key both contributed to changing $\Delta$; the trace does not identify
one projection as the sole driver. The proposed claim that length 100 always
grows raw margin first is not supported: the first 200-step raw-gap changes
had seed-dependent signs, and later query/key gains were not uniformly larger
for length 100. The better-supported description is a later difference in
realized coefficient growth.

Seed 5 exposes the initialization-sensitive branch. Its *initial* length-100
positive target mass fell from 0.004964 at baseline to 0.003702 at $c_0=0.1$
and 0.000217 at $c_0=1.0$, with identical initial query/key/head weights.
After 200 `[10]`-training updates, the corresponding common length-100 masses
were about 0.000807, 0.000214 and 0.000000426. In the 200-update diagnostics,
the head slope first became positive at 800 (baseline) and 2000 ($c_0=0.1$)
for `[10]`, while its raw margin first became positive at 1400 and 3000;
for `[100]` at $c_0=0.1$, these observations shifted to 4800 and 6200.
At $c_0=1.0$, neither head slope nor raw margin became positive by 6400 for
either training length. This sequence is consistent with suppressed positive
attention delaying or trapping the transition to a fitted solution. Because
changing $c_0$ also changes initial attention and gradients, it does not
isolate attention suppression as a unique cause. See the [early transitions](../../runs/training_length/seed_replication_20260929_01/analysis/trace_early_transitions.csv),
[window sums](../../runs/training_length/seed_replication_20260929_01/analysis/trace_window_summary.csv)
and [per-update trace](../../runs/training_length/update_trace_20260929_01/learned_log_n10_s5_c1/update_trace.csv).

## Validation and limits

The 15 new-seed, 20 intervention and 18 trace directories have complete status,
source/environment manifests, 200-update diagnostics and realized exposure:
[replication manifest](../../runs/training_length/seed_replication_20260929_01/manifest.json),
[intervention manifest](../../runs/training_length/coefficient_init_20260929_01/manifest.json),
and [trace manifest](../../runs/training_length/update_trace_20260929_01/manifest.json).
The trace probe and all full replays passed exact weight/history/diagnostic
equality. Eight focused follow-up tests passed; the 11 Stage 3 evaluation tests
and eight Stage 4A tests passed (the Stage 4A temporary-directory test used a
fixed writable workspace path because the sandbox denied its usual temporary
directory). All final explicit sequence evaluations were limited to lengths
10, 100, 1000 and 10000. Failure-length ranks beyond that grid remain
head-aware closed-form predictions, not measured 10M behavior. The analysis
[audit](../../runs/training_length/seed_replication_20260929_01/analysis/audit.json)
and [source snapshot](../../runs/training_length/update_trace_20260929_01/source_snapshot/)
preserve the checks; outputs under `runs/` remain local and ignored.

The most targeted next control is to match the initial effective margin at a
declared reference length while varying $c_0$. It must still account for the
changed query/key parameterization and initial head/other-length attention;
no such additional control was part of this experiment.
