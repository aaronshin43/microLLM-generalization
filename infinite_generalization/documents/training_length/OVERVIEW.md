# Training-Length Follow-up: Research Overview

Last updated: October 10, 2026. This document summarizes the research flow through the coefficient-learning-rate interventions and independent-seed predictive check. See the linked result documents for detailed settings, measurements and limitations.

## Starting question

The [summer final report](../FINAL_REPORT.md) studied length-aware scaling in a reduced binary target-presence detector trained at length 10. The follow-up began by asking:

> How do training length and its distribution affect the attention-scaling coefficient, score margin and length generalization selected by optimization?

The detector and task stayed fixed: one target type, one non-target type, fixed one-hot values, a final non-target query and a linear binary head. Learned-log was retained as a diagnostic of parameter selection, with $\alpha(n)=1+c\log(1+n)$. The model learns a coefficient within a supplied logarithmic form; it does not discover that functional form. Counting remains an optional extension, and parity is outside this follow-up.

## How the experiments developed

### 1. Small pilot: compare training lengths

The [pilot](PILOT_RESULTS.md) compared training at `[10]`, `[100]` and `[10,100]` across constant, fixed-log and learned-log scaling, using three paired seeds and 1600 updates per run.

All conditions fitted their training classes. Learned-log selected different combinations of $c$ and $\Delta$ across lengths. The mixture had a slightly larger exponent $c\Delta$ than `[100]`, but a smaller raw margin and an earlier predicted failure. Relative to `[10]`, it instead had a larger raw margin and a smaller exponent. This showed that the reference condition matters and that exponent alone cannot rank finite extrapolation.

**Next question:** do these patterns persist with more optimization?

### 2. Longer-budget confirmation: track the learning path

The [6400-update confirmation](CONFIRMATION_6400_RESULTS.md) restarted the nine learned-log conditions from their original initializations and exactly reproduced their 1600-update pilot prefixes.

Only `[10]` crossed $c\Delta>1$ in all three seeds. The mixture's small exponent advantage over `[100]` persisted, and analytical failure-length rankings changed during training. Parameter differences also remained at matched common length-100 loss, weakening an explanation based solely on a scalar training-speed difference.

**Next question:** is this reproducible, and how much does coefficient initialization matter?

### 3. New seeds and coefficient interventions

The [mechanism follow-up](MECHANISM_FOLLOWUP_RESULTS.md) added seeds 3–7 and tested larger initial coefficients for the single-length conditions.

At the original small coefficient, all eight seeds showed the exponent ordering `[10] > [10,100] > [100]` at 6400 updates; only `[10]` crossed 1. The full analytical failure-length ranking was less consistent. Starting at $c_0=0.1$ allowed both single lengths to fit and cross in all five new seeds. At $c_0=1$, seed 5 failed to fit positives at both lengths. Larger coefficients also reversed the cross-length raw-margin ordering among fitted runs.

**Next question:** does changing the starting attention explain this initialization sensitivity?

### 4. Realized-update traces: locate the difference

The same [mechanism follow-up](MECHANISM_FOLLOWUP_RESULTS.md) traced actual AdamW updates. Early query/key contributions to exponent changes were larger in magnitude but had seed-dependent signs. In sampled middle-training windows, coefficient changes contributed more, and their contribution was consistently larger for `[10]` than `[100]` in the traced seeds.

This supported a later difference in realized coefficient growth rather than a universal claim that length-100 training first increases the raw margin. The failed seed showed suppressed target attention and delayed attention/head transitions. Raw gradient magnitude alone did not describe the applied updates.

**Next question:** will matching initial predictions remove these learning-path differences?

### 5. Match initial behavior and test query/key factorization

The [matched-initial-behavior controls](MATCHED_INITIAL_BEHAVIOR_RESULTS.md) varied $c_0$ while rescaling query/key to preserve initial attention, logits and loss at a declared reference length. Own-training-length controls were included for both single lengths.

Coefficient sensitivity remained despite matched starting behavior. Seed 5 still failed at $c_0=1$ for both lengths, and its matched length-100 run at $c_0=0.1$ also failed within the budget. Among fitted runs, parameter differences remained at comparable loss. Actual first updates differed because the transformed models had different parameter sensitivities.

The conditional factorization check then represented the same initial scoring function with symmetric, query-only and key-only scaling. Early updates changed, but fitting outcomes stayed the same in the tested seeds. This demonstrated sensitivity of the dynamics to factorization without showing an outcome change for these transforms.

### 6. Branch after successful fitting: intervene on coefficient learning rate

The [coefficient-rate branch experiment](COEFFICIENT_LR_BRANCH_RESULTS.md) replayed six baseline parents to update 1600, then cloned their full model, AdamW and data/RNG states into coefficient-rate multipliers 1/3, 1 and 3. Training lengths were 10 and 100, with seeds 3, 4 and 6. Query/key and head learning rates stayed fixed; all 18 paths ended at total update 6400. The multiplier-1 branches exactly reproduced the historical baselines.

All paths retained fitting. Exponents followed fast > control > slow at equal updates and common length-100 BCE 0.01. The absolute intervention effect was larger at length 10 in every paired seed, widening the length gap under faster coefficient learning and shrinking it under slower learning. Final sibling losses and heads were very similar despite large exponent differences. Smaller raw margins partly compensated for faster coefficient growth but did not remove its effect.

Approximately 90–91% of the control's signed post-1600 exponent-gap growth fell in the coefficient term of the exact update decomposition. This is accounting, not a causal mediation percentage. The stronger evidence is the predeclared length-specific intervention response and its persistence at common loss. The experiment addresses selection after fitting, not the origin of the parent differences before update 1600.

**Next question:** can a previously untested coefficient rate be predicted without adjusting the forecast after seeing its result?

### 7. Held-out rate: test multiplier 2

The [multiplier-2 check](COEFFICIENT_LR_MULTIPLIER2_RESULTS.md) reused the same six fitted parents. A phenomenological power-response model and ±20% effect bounds were frozen using the existing rate results before the new continuations.

All 12 numerical predictions passed across the 6400 endpoint and common BCE 0.01; fitting and all directional predictions were retained. The maximum relative effect error was 13.39%. All effects exceeded their forecasts, showing systematic underprediction despite passing the tolerance. This is interpolation to an untested rate on the same parents, not independent-initialization evidence or a derived optimization law.

**Next question:** does a predictive response rule transfer to new initializations?

### 8. Independent seeds: transfer the response rule and inspect Adam history

The [new-seed check](COEFFICIENT_LR_NEWSEEDS_RESULTS.md) used seeds 8 and 9 at lengths 10 and 100, branching eligible 1600-update states into rates 1 and 2. All four parents were eligible and no seed was replaced. A separate normalized-effect rule, calibrated on seeds 3/4/6 and frozen before new training, used each new control's post-branch exponent growth to predict its multiplier-2 effect. Forecasts were saved before any multiplier-2 continuation. This is a conditional prediction given the control trajectory, not a prediction of that trajectory from initialization.

All four primary endpoint forecasts and four common-loss forecasts passed the unchanged ±20% effect bounds. The largest error was 7.65%, again with underprediction in every comparison. All paths retained fitting, and length 10 again responded more strongly. The result supports transfer over these two new seeds without fitting new per-parent rate curves.

Read-only first-update analysis across five old/new seeds identified a local explanation: length 10 had smaller raw coefficient gradients and raw margins, but larger softplus sensitivity and Adam normalized steps. The length-100 coefficient second moment was approximately 7.6–9.2 times larger. These factors reconstructed the approximately 2.2–2.4-fold first-response ratio. Making the two denominators equal only in the arithmetic check reversed the local ordering; this was not a trained optimizer-state intervention.

For the available seed-3/4 traces, updates 1–200 supplied at most 0.0031% of the second moment at 1600. The earliest gradients therefore do not explain the observed denominator difference in those cells; its history lies predominantly in updates 201–1600, which lack complete detailed tracing. The origin of that history and its causal influence on the later path remain unresolved.

## Current position

The research question has narrowed from training-length effects to:

> Among solutions that successfully fit the training task, why do training length and optimization conditions select different length-generalization exponents?

The post-fitting interventions show that coefficient learning rate can change exponent selection and the length contrast while preserving fitting and nearly identical measured performance. A response rule transferred to two independent seeds. Softplus sensitivity and Adam normalization explain the immediate length-dependent response locally; they do not yet explain the complete training trajectory or how the earlier moment differences formed.

Initialization-sensitive fitting failure is a related but separate question. Reference-matched controls and a local loss derivative support a coupled attention/head transition bottleneck in the failed case, without isolating its unique cause. Seed-5 rescue is not the central direction of the exponent-selection study.

Important boundaries:

- Results apply to the reduced detector, tested seeds, transforms and finite training budgets. Continued recovery or eventual convergence remains unresolved.
- Matching behavior at one reference length does not match all lengths, parameter norms, Jacobians or optimizer dynamics.
- The factorization checks match the initial scoring function at every length, but test only three specific factorizations.
- Explicit sequence evaluation reaches length 10000. Longer-length failure estimates and no-finite-crossing conclusions are head-aware analytical predictions for fixed trained parameters, not measured 10M results.
- A larger exponent does not automatically imply better classification. Failed fits must remain visible, and fitted runs with no finite predicted failure need not have a strict failure-length ordering.
- Post-fitting branch effects are conditional on existing model geometry and optimizer history at update 1600. They do not causally explain the entire earlier training-length difference.
- Common-loss matching is descriptive and does not match all head, geometry or optimizer states. Exact update decompositions are accounting identities, not causal mediation estimates.
- The held-out-rate power fit and independent-seed normalized-effect rule are different empirical predictors. Their tolerances are declared test criteria, not confidence intervals. Both systematically underpredicted the measured effects.
- The independent-seed rule requires the new control trajectory. Its success does not establish an initialization-only forecast or a universal optimizer law.
- Equalizing Adam denominators was a local arithmetic check only. Early-history reconstruction covers seeds 3 and 4, and does not reconstruct the full 201–1600 interval.
- The existing trials identify candidate mechanisms; broader relevance and novelty still need assessment.

## Remaining questions and navigation

The next useful direction is to explain the length-dependent coefficient response through geometry and optimizer history, rather than adding more rate choices. A scoped replay recording the coefficient-gradient history through updates 201–1600 could locate how the second-moment difference forms. A subsequent optimizer-history intervention would need explicit controls and frozen predictions to test its effect after fitting. These are candidates, not approved launch specifications.

Retain the original question: can the explanation predict new conditions, move the selected generalization solution in an expected direction, explain several successful seeds, and survive a modest relaxation of model assumptions? Review the direction after a limited control; do not expand experiment families automatically.

- [PLAN.md](PLAN.md): research plan and direction-review criteria; consult newer launch specifications for finalized settings.
- [DETECTION_GUIDE.md](DETECTION_GUIDE.md): model and code walkthrough.
- [BASELINE_REPRODUCTION.md](BASELINE_REPRODUCTION.md): historical artifacts and reproduction paths.
- [COEFFICIENT_LR_BRANCH_RESULTS.md](COEFFICIENT_LR_BRANCH_RESULTS.md): post-fitting rate intervention and predeclared length-response tests.
- [COEFFICIENT_LR_MULTIPLIER2_RESULTS.md](COEFFICIENT_LR_MULTIPLIER2_RESULTS.md): held-out-rate forecast validation.
- [COEFFICIENT_LR_NEWSEEDS_RESULTS.md](COEFFICIENT_LR_NEWSEEDS_RESULTS.md): independent-seed forecasts, local Adam analysis and unresolved history.
- The earlier result documents linked above: training-length, initialization and behavior-matching evidence.

Keep this overview brief and update it after major experiments. Detailed numbers and new experiment specifications belong in their dedicated documents. Run artifacts linked from those documents remain local and ignored unless explicitly archived for sharing.
