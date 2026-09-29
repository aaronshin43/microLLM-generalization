# Detection Training-Length Pilot Results

All **27 runs completed** at 1600 optimizer updates each. Sweep wall time was
177.1 seconds (2.95 minutes), below the user's
60-minute cap. Exact settings and predictions written before launch are in
[PILOT_SPEC.md](PILOT_SPEC.md); the user's research plan remains in [PLAN.md](PLAN.md).

## Main comparison

For learned-log, mixture minus single-length 100 changes raw margin by -0.620 ± 0.023, coefficient by 0.0058 ± 0.0014, and exponent by 0.015 ± 0.011. These are paired differences across the three seeds, not independent-group estimates.

27/27 runs classified both classes correctly at all of their training lengths.
This is a fitting check, not evidence that optimization has converged. All seed-level
conditions are reported below; no longer-budget confirmation has been performed.

Values are means ± sample standard deviations across seeds 0, 1, 2. `Positive passes`
counts seeds with measured positive accuracy 1 at length 10000. Failure ranges give
$\log_{10} n^*$ from a **closed-form prediction**, not explicitly evaluated failure lengths.
An `n/a` entry means the descending finite threshold solver does not apply; inspect
the head and scaling regime rather than treating this as unconditional success.

| Mode | Training lengths | Final val BCE | $\Delta$ | $c$ | $c\Delta$ | Head threshold | Positive passes at 10000 | Predicted $\log_{10} n^*$ range |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| constant | [10] | 0.0352 ± 0.0025 | 8.901 ± 0.273 | n/a | n/a | 0.497 ± 0.001 | 0/3 | 3.75–3.99 |
| constant | [100] | 0.0312 ± 0.0020 | 11.275 ± 0.184 | n/a | n/a | 0.493 ± 0.003 | 3/3 | 4.83–4.98 |
| constant | [10,100] | 0.0360 ± 0.0026 | 10.554 ± 0.230 | n/a | n/a | 0.495 ± 0.002 | 3/3 | 4.50–4.69 |
| log | [10] | 0.0384 ± 0.0031 | 4.377 ± 0.196 | n/a | n/a | 0.498 ± 0.001 | 3/3 | n/a |
| log | [100] | 0.0383 ± 0.0028 | 2.885 ± 0.093 | n/a | n/a | 0.497 ± 0.002 | 3/3 | n/a |
| log | [10,100] | 0.0395 ± 0.0029 | 4.006 ± 0.193 | n/a | n/a | 0.498 ± 0.002 | 3/3 | n/a |
| learned_log | [10] | 0.0353 ± 0.0025 | 8.014 ± 0.297 | 0.0729 ± 0.0060 | 0.583 ± 0.027 | 0.497 ± 0.001 | 3/3 | 8.18–8.67 |
| learned_log | [100] | 0.0317 ± 0.0020 | 9.463 ± 0.218 | 0.0572 ± 0.0026 | 0.541 ± 0.013 | 0.493 ± 0.003 | 3/3 | 8.89–9.14 |
| learned_log | [10,100] | 0.0361 ± 0.0026 | 8.843 ± 0.237 | 0.0629 ± 0.0029 | 0.556 ± 0.011 | 0.495 ± 0.002 | 3/3 | 8.60–8.73 |

All final negative predictions at length 10000 were correct in all 27 runs.
For fixed-log the effective growth exponent is $\Delta$, not a learned coefficient.
Constant scaling always has asymptotic target mass 0 for fixed finite weights.
For learned-log, assess $c\Delta$ together with the head; passing a finite benchmark
does not imply convergence of target mass to 1.

![Learned-log optimization trajectories](../../runs/training_length/pilot_20260928_01/sweep/analysis/learned_log_trajectories.png)

The trajectories use exact two-score analytical diagnostics at initialization and
every 200 updates. Shading is one sample standard deviation across three seeds.
Training/validation losses in each run's history are measured through the original
training loop; analytical diagnostic BCE is labeled separately.

## Controls and realized exposure

### Interpretation against the pre-launch prediction

The same-maximum comparison did **not** show the tentative pattern of a mixture
raising raw margin while lowering the exponent relative to `[100]`. In each paired
seed, the mixture had a smaller $\Delta$, larger $c$, and slightly larger $c\Delta$.
Its analytical failure length was nevertheless **earlier** than `[100]` in every seed:
the smaller constant margin and head threshold also matter at finite lengths.
This is a useful warning against ranking trained classifiers by $c\Delta$ alone.

Relative to `[10]`, the learned-log mixture had a larger raw margin but a smaller
coefficient and exponent in each seed, consistent with the direction of the older
Stage 3B comparison. The choice of single-length reference changes the observation;
the same-maximum comparison is essential for interpreting a mixture.

All nine learned-log runs have $c\Delta<1$ and positive head thresholds, so their
exact two-score analytical predictions imply eventual positive-class collapse even
though they pass the explicit grid. All fixed-log runs have $\Delta>1$ and heads
that distinguish the limiting positive/negative representations. These conclusions
are scoped to fixed trained parameters in this reduced architecture.

The losses and trajectories are still changing at the end of 1600 updates. A
matched longer-budget comparison is needed before calling the observed parameter
selection stable. The paired exponent increase versus `[100]` is small, and three
seeds cannot support a broad claim about multi-length training.

### Exposure audit

Every run processed exactly 102400 sequences, 51200 positive and 51200 negative.
Mixtures processed 51200 sequences at each length, balanced within each length.
Token totals per run were 1024000 (`[10]`), 10240000 (`[100]`) and 5632000 (`[10,100]`).
Full initial state hashes match across every condition for each paired seed,
including across modes. Model, optimizer, learning rate and dataset reuse were fixed.
The mixture has half the length-100 exposure of `[100]`, so it isolates neither
token budget nor length-100 sample count; account for this in mechanism claims.

Explicit sequence evaluation used lengths 10, 100, 1000 and 10000, with 25 positive
and 25 negative examples at each length. Actual forward target masses agreed with
independent two-score calculations within `rtol=3e-5`, `atol=3e-6` in all 108
run/length comparisons. This tolerance permits float32 reduction rounding at the
modest explicit lengths and is not a 10M validation claim.

Beyond this grid, `analytical_predictions.csv` contains closed-form predictions
through $10^{30}$. No 10M sequence was generated or measured. Failure estimates
use the trained head threshold, assuming the exact two-token model and positive
head slope. Intermediate optimizer states were not saved; future longer-budget
comparisons must restart unless faithful resume support is added.

## Per-seed results

Positive accuracy is measured at length 10000; predicted failure is $\log_{10} n^*$.

| Run | Final val BCE | $\Delta$ | $c\Delta$ | Head threshold | Positive accuracy | Predicted failure |
|---|---:|---:|---:|---:|---:|---:|
| constant_n10_s0 | 0.0376 | 9.1696 | n/a | 0.4978 | 0 | 3.99 |
| constant_n100_s0 | 0.0331 | 11.4491 | n/a | 0.4939 | 1 | 4.98 |
| constant_n10_100_s0 | 0.0386 | 10.7862 | n/a | 0.4959 | 1 | 4.69 |
| constant_n10_s1 | 0.0355 | 8.9103 | n/a | 0.4979 | 0 | 3.87 |
| constant_n100_s1 | 0.0314 | 11.2929 | n/a | 0.4951 | 1 | 4.91 |
| constant_n10_100_s1 | 0.0360 | 10.5489 | n/a | 0.4962 | 1 | 4.59 |
| constant_n10_s2 | 0.0326 | 8.6239 | n/a | 0.4955 | 0 | 3.75 |
| constant_n100_s2 | 0.0292 | 11.0819 | n/a | 0.4889 | 1 | 4.83 |
| constant_n10_100_s2 | 0.0334 | 10.3272 | n/a | 0.4924 | 1 | 4.50 |
| log_n10_s0 | 0.0404 | 4.5326 | n/a | 0.4987 | 1 | n/a |
| log_n100_s0 | 0.0399 | 2.9539 | n/a | 0.4969 | 1 | n/a |
| log_n10_100_s0 | 0.0416 | 4.1657 | n/a | 0.4983 | 1 | n/a |
| log_n10_s1 | 0.0400 | 4.4406 | n/a | 0.4991 | 1 | n/a |
| log_n100_s1 | 0.0400 | 2.9219 | n/a | 0.4989 | 1 | n/a |
| log_n10_100_s1 | 0.0408 | 4.0605 | n/a | 0.4991 | 1 | n/a |
| log_n10_s2 | 0.0348 | 4.1574 | n/a | 0.4971 | 1 | n/a |
| log_n100_s2 | 0.0350 | 2.7799 | n/a | 0.4951 | 1 | n/a |
| log_n10_100_s2 | 0.0361 | 3.7912 | n/a | 0.4961 | 1 | n/a |
| learned_log_n10_s0 | 0.0377 | 8.3035 | 0.5627 | 0.4980 | 1 | 8.25 |
| learned_log_n100_s0 | 0.0336 | 9.6898 | 0.5254 | 0.4944 | 1 | 8.89 |
| learned_log_n10_100_s0 | 0.0387 | 9.0818 | 0.5476 | 0.4963 | 1 | 8.73 |
| learned_log_n10_s1 | 0.0357 | 8.0268 | 0.5733 | 0.4982 | 1 | 8.18 |
| learned_log_n100_s1 | 0.0319 | 9.4440 | 0.5506 | 0.4956 | 1 | 9.14 |
| learned_log_n10_100_s1 | 0.0361 | 8.8381 | 0.5528 | 0.4968 | 1 | 8.60 |
| learned_log_n10_s2 | 0.0327 | 7.7105 | 0.6131 | 0.4959 | 1 | 8.67 |
| learned_log_n100_s2 | 0.0296 | 9.2560 | 0.5460 | 0.4896 | 1 | 8.89 |
| learned_log_n10_100_s2 | 0.0335 | 8.6089 | 0.5678 | 0.4930 | 1 | 8.68 |

## Artifacts and validation

- [Run manifest and environment](../../runs/training_length/pilot_20260928_01/sweep/manifest.json): exact configs, revision,
  dirty state, device and launch arguments; `source_snapshot/` preserves the code used.
- [All 27 run summaries](../../runs/training_length/pilot_20260928_01/sweep/pilot_summary.csv), [condition summary](../../runs/training_length/pilot_20260928_01/sweep/analysis/condition_summary.csv)
  and [paired differences](../../runs/training_length/pilot_20260928_01/sweep/analysis/paired_differences.csv).
- [Realized exposure](../../runs/training_length/pilot_20260928_01/sweep/analysis/realized_exposure.csv) and each run's
  `exposure.json`, `training_diagnostics.csv`, `train_history.csv`, `model.pt`,
  `metrics_by_length.csv` and diagnostic CSVs.
- New observer/failure-solver tests: 2 passed; existing Stage 3 tests: 11 passed;
  Stage 4 tests: 34 passed after the single temporary-directory environment failure
  was rerun with workspace storage. The original Stage 4A test failure was a write
  permission error, not a model assertion failure. CPU observation preserved weights,
  RNG state and history; a 200-update CUDA comparison preserved final weights/history.

Current launch revision: `b970ea53aa2680ce78945eb4f98522ab0a2e6425`, with recorded source
changes in the manifest/snapshot. Python 3.12.10, PyTorch
2.12.0+cu126, device `cuda`,
GPU NVIDIA GeForce RTX 4060 Laptop GPU. Existing historical artifacts were preserved.
Run outputs remain ignored under the repository policy, so these artifact links
are local and need explicit archiving for use on another machine.

## Review before another launch

This three-seed, one-budget pilot is exploratory. Compare paired effects and their
trajectories with final loss before selecting a 6400-update matched confirmation.
Mixture advantages confined to short finite benchmarks, a lower coefficient by
itself, or training accuracy alone do not establish a new generalization mechanism.
Confirm a selected pattern with matched longer training and additional seeds before
expanding lengths or claiming novelty. No confirmation or intervention was launched.
