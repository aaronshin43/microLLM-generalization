# Training-Length Follow-up: Detection

## Purpose and fixed task

Compare how training length and its distribution affect calibration in the reduced **binary target-presence detector** from the [final report](../FINAL_REPORT.md). Reuse `src/stage3_simplified_attention.py`; keep one target type, one non-target type, fixed one-hot values, the final non-target query and a linear binary head. Positives contain exactly one target; negatives contain none. The fixed first-position convention does not make this FIRST classification.

Start with [DETECTION_GUIDE.md](DETECTION_GUIDE.md), then [BASELINE_REPRODUCTION.md](BASELINE_REPRODUCTION.md). This document proposes candidates, not a finalized sweep specification. Counting is an optional later extension; parity is outside the current scope. The root `TODO.md` retains an earlier semester plan centered on detection/counting. It was read but not rewritten; this cleanup's immediate priority is detection.

## Role of learned-log

Learned-log is not required to solve detection and is not proposed as a new scaling method. Constant and fixed-log remain essential reference conditions. Retain learned-log as a diagnostic of **which length-dependent solution optimization selects**, rather than assuming it must outperform fixed-log.

For the current parameterization,

```math
\alpha(n)\Delta=\Delta+(c\Delta)\log(1+n).
```

At a single training length, different combinations of $c$ and $\Delta$ can produce the same effective margin while implying different long-length behavior. The functional form is supplied by the researcher; the model learns its coefficient, not the logarithmic law itself. Multiple lengths provide additional constraints, but classification labels do not guarantee a unique decomposition or a generalizing solution.

Track the effective margin, exponent and head together. Do not treat an increase in $c$ alone as improvement, or $c\Delta>1$ as a necessary condition for all successful detection: the equality case depends on the head. Fixed-log still learns a score margin, so fixing its multiplier does not fix its effective exponent.

If learned-log adds no explanatory result beyond initialization sensitivity or the existing closed form, reduce its role instead of expanding its sweep automatically. Comparisons of coefficient initialization should distinguish changing the initial effective attention from changing its parameterization; matched effective-margin controls can be a later targeted intervention.

## Candidate conditions and controls

Candidate single lengths are 10, 100 and 1000. Candidate mixtures include `[10,100]` and `[10,100,1000]`; other mixtures can be chosen after reviewing the budget. Compare `[10,100]` with `[100]`, and `[10,100,1000]` with `[1000]`, to hold maximum training length fixed while examining diversity. Keep constant and fixed-log baselines available when interpreting learned-log behavior.

Control or explicitly report:

- Optimizer updates, total processed sequences, initialization, class proportions, architecture, optimizer and learning rate. `train_examples` is per length, so equal epochs do not imply equal budgets.
- Final partial batches: 2000 examples at batch 64 produces a 16-example last batch. Equal steps alone therefore may not imply equal processed sequences. Choose an explicit allocation rule before launch; do not silently replace the report baseline settings.
- Total processed tokens separately, since equal sequence counts at different lengths imply different computation and exposure.
- Seed pairing and length-list order. Current loader seeds depend on list offset, and all batch lists are shuffled per epoch. Verify identical initial states for comparisons rather than assuming seed alone controls every part of the data path.
- Fixed dataset reuse versus fresh per-epoch sampling. Current code reuses each generated dataset; changing this would be a separate experiment.

## Measurements and existing coverage

Record training/validation loss, $c$, raw $\Delta$, $c\Delta$, head slope and threshold, and attention mass/logits/predictions by evaluation length. Compare finite accuracy with the asymptotic two-score prediction, including head behavior at the boundary. Keep explicit sequence evaluation distinct from closed-form extrapolation.

Current code writes epoch loss/accuracy in `train_history.csv` and final diagnostics in `metrics_by_length.csv`; head weights/bias and learned coefficient are already present. Threshold can be derived from the head. It saves only a final checkpoint, not a trajectory of $c$, $\Delta$ and threshold. If trajectories are required, design a separately reviewed small diagnostic extension before starting; no recorder or sweep runner was added in this cleanup. Report consumed sequences/tokens explicitly because those cumulative counts are not currently recorded.

Include each training length in the eval list if relying on `mean_theory_target_attention_using_train_delta`; the helper otherwise falls back to the first evaluation row. The analyzer prefers length 10 and only analyzes one target/non-target pair. The two-token follow-up can reuse its weight decomposition, but should calculate threshold and length-aware diagnostics explicitly.

## Existing Stage 3B evidence and questions

[Stage 3B](../STAGE3_SIMPLIFIED_LENGTH_AWARE_ATTENTION.md#stage-3b-multi-length-training-negative-result) already explored multi-length detection. Existing artifacts are under `runs/stage3b/`; their directory presence was checked, but they were not retrained during cleanup. In those tested conditions, broader length mixtures increased raw margin while keeping the learned coefficient low, leaving $c\Delta<1$. Treat this as historical empirical evidence rather than a general claim that mixtures cannot help.

Questions for the follow-up:

- At controlled update and sequence budgets, does changing the single training length select different combinations of $c$ and $\Delta$?
- Compared with a single length at the same maximum, does a mixture alter calibration or mainly increase the raw margin?
- Does sufficient optimization change that pattern, and how sensitive is it to seed/initialization?
- Does finite benchmark success persist because of attention growth, a large constant margin, or the learned head threshold?

## Short exploration, then a direction decision

The first experiment is a bounded pilot, not a commitment to run every condition or a guarantee of a publishable contribution. Start experiments alongside focused reading; do not wait for a complete literature review. Before launching, write down the expected pattern and what would contradict it.

**Confirmed direction (2026-09-28):** the user chose to start with the small pilot. The first training-length comparison is `[10]`, `[100]` and `[10,100]`; length-1000 training conditions are deferred until the pilot review. The proposed modes, seed count, step budget and diagnostic cadence below remain recommendations, not a fully finalized launch specification. This decision records the research scope; no experiment was launched by updating this plan.

Sequence, with execution details still pending below:

1. **Pilot:** compare `[10]`, `[100]` and `[10,100]` with constant, fixed-log and learned-log, using three paired seeds. This is 27 runs at one initial step budget. Check one short run for runtime before estimating the total budget. Defer the length-1000 conditions until this first comparison is reviewed.
2. **Inspect:** examine losses, margins, head thresholds and predicted failure lengths as well as accuracy. Report every pilot condition, including null results; do not select a narrative from the best seed.
3. **Confirm:** repeat a candidate pattern with additional seeds and a matched longer budget for the relevant single/mixed comparison. A difference caused solely by undertraining is a different finding from a stable selection effect.
4. **Intervene:** choose one explanation and test it, for example through coefficient initialization or a controlled scoring parameterization. Separate these changes from the original training-length comparison.
5. **Decide:** deepen the mechanism analysis and relax one model assumption only if the result warrants it. Otherwise document the outcome and reconsider the question rather than adding more lengths and seeds indefinitely.

At the first review, ask:

- Is the pattern consistent across seeds and distinguishable from budget, batch-allocation or evaluation artifacts?
- Does it reveal something beyond the already-known dilution formula or the historical Stage 3B observation?
- Can it generate a prediction for an untested condition or an intervention that would change the outcome?
- Is there a focused next experiment that could falsify the proposed explanation?

A promising direction might explain why a mixture improves a finite benchmark while selecting a weaker asymptotic exponent, or why equivalent initial behavior leads to different learned solutions. These are candidate findings, not established results. Their novelty still requires comparison with prior work. A reduced model can support a scoped mechanism study, but wider claims require additional theory or validation beyond its restrictive assumptions. A completed parameter sweep alone is not the contribution.

## Decisions needed before experiments

The small-pilot scope is confirmed. **The remaining numerical defaults are recommendations, not a finalized experiment specification.** Resource budget and depth of diagnostics remain to be settled. The implementing agent can resolve routine execution details within those choices and record the exact settings before launch.

| Decision | What it determines / tradeoff | Suggested pilot starting point |
|---|---|---|
| Training-length scope — confirmed | Screen the small contrast before considering the full five-condition comparison. | Start with `[10]`, `[100]`, `[10,100]`; consider `[1000]` and `[10,100,1000]` after review. |
| Scaling modes | Whether the question concerns general training-length effects or only learned-log behavior. | All three modes in the small pilot; use learned-log as a diagnostic, not the assumed winner. |
| Seed count and pairing | How much random initialization variability can be assessed. More seeds increase cost; three seeds are exploratory evidence only. | Seeds 0, 1, 2, paired across training-length conditions with verified identical initial model states for each mode. Use additional seeds for confirmation. |
| Optimizer-step budget | Separates equal-compute-procedure comparisons from differences in training duration. Perfect accuracy does not imply optimization has stopped. | Initially 1600 updates, with 6400-update follow-up for selected matched comparisons. Keep batch size, optimizer and learning rate fixed; do not early-stop each condition at a different accuracy. |
| Class and length allocation | Defines the training objective and determines whether equal steps also mean equal processed sequences. | Balanced positive/negative examples at each length and equal length exposure within mixtures. Target batch size 64 and explicitly specify partial-batch handling. Preserve the report's 2000-example setting in historical reproduction; a divisible-size pilot dataset, such as 2048 examples per length, is a separate documented choice. Audit realized sequence counts and class/length exposure; equal datasets do not guarantee exact balance in a truncated epoch. |
| Dataset reuse | Fresh sampling changes the experiment and RNG path. In this fixed two-token task, repeated examples are not new semantic input diversity. | Keep the current fixed-dataset reuse for the first comparison. |
| Evaluation lengths | Defines the measured extrapolation range; explicit evaluation becomes expensive at large lengths. | Include all training lengths plus a modest log-spaced explicit grid, e.g. 10, 100, 1000, 10000. Add larger explicit checks near a predicted transition only when affordable. Label more distant closed-form evaluations as analytical predictions. |
| Diagnostic/checkpoint cadence | Final-only diagnostics show where training ended; trajectories show how it got there. Saving optimizer state is needed for a faithful resumed continuation. | Prefer lightweight diagnostics every 200 optimizer updates and at the final update, plus the initial state. This requires a small recorder change. Full intermediate checkpoints are optional; record whether longer-budget runs restart or faithfully resume. If avoiding code changes initially, use final-only diagnostics and limit claims about learning dynamics. |
| Device and resource cap | Determines feasible pilot size, reproducibility conditions and whether long-sequence checks are practical. | Use one available device consistently; time a representative run and agree on a wall-time cap before launching the pilot. Do not assume the GPU is available or needed. |
| Explicit 10M validation | Tests the actual sequence implementation but is unnecessary for every pilot run in the exact two-score setting. | Defer it. After a useful pattern is confirmed, choose representative checkpoints for chunked explicit validation. A short check or a closed-form prediction is not a measured 10M result. |

No new sweep or 10M evaluation has been launched as part of this plan. These decisions do not block understanding or short verification of the existing baseline, but the allocation rule, budget and recording behavior must be explicit before making a fair experimental comparison.

## Prepared locations and remaining work

```text
src/stage3_simplified_attention.py                    existing implementation
documents/training_length/DETECTION_GUIDE.md          model and execution guide
documents/training_length/BASELINE_REPRODUCTION.md    provenance and verified commands
documents/training_length/PLAN.md                     this follow-up proposal
runs/training_length/cleanup_checks/20260928_detection_cleanup_01/
                                                     audit and short validation output
runs/training_length/<new-experiment-id>/             future results, existing ignore policy
configs/local/                                      temporary supported configs only
```

Stage 3 currently supports CLI arguments only. Use documented CLI specifications instead of inventing YAML flags; a shared `configs/training_length/` directory is unnecessary until a supported file format is actually needed.

Completed preparation: report artifacts mapped for all 40 runs; current model/loader/metric semantics documented; 11 Stage 3 tests and three short pipelines passed; independent positive/negative two-score calculations passed. Existing outputs and model behavior were preserved. No source code changes were needed.

Remaining blockers for exact historical reproduction are missing historical revision/environment and missing checkpoints in the older `stage3base` family. Local report-family checkpoints do exist. There is no blocking implementation issue identified for the scoped two-token follow-up; trajectory diagnostics and fair sequence/token accounting require an explicit specification before launch.

Possible later cleanup: portable report figure paths, checkpoint-driven mechanism figure generation, generalizing analyzer length selection, historical artifact archiving, input-validation coverage, and optional intermediate diagnostics. Keep these separate from the current documentation cleanup and from changes that would affect RNG order, architecture or training behavior. Stage 4A/4B reuse Stage 3 model/helpers, so future shared code changes should include their relevant regression tests.
