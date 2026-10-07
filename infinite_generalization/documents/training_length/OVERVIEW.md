# Training-Length Follow-up: Research Overview

Last updated: October 6, 2026. This document summarizes the research flow through the matched-initial-behavior experiments. See the linked result documents for detailed settings, measurements and limitations.

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

## Current position

The research question has narrowed from training-length effects to:

> Why can models with matching initial predictions follow different optimization paths, and how do those paths affect fitting and length generalization?

The experiments establish reproducible parameter selection in the original setup, substantial initialization sensitivity, and learning-path differences that survive reference-length behavior matching. Traces and a local loss derivative support a coupled attention/head transition bottleneck in the failed case. They do not yet isolate its unique cause.

Important boundaries:

- Results apply to the reduced detector, tested seeds, transforms and finite training budgets. Continued recovery or eventual convergence remains unresolved.
- Matching behavior at one reference length does not match all lengths, parameter norms, Jacobians or optimizer dynamics.
- The factorization checks match the initial scoring function at every length, but test only three specific factorizations.
- Explicit sequence evaluation reaches length 10000. Longer-length failure estimates and no-finite-crossing conclusions are head-aware analytical predictions for fixed trained parameters, not measured 10M results.
- A larger exponent does not automatically imply better classification. Failed fits must remain visible, and fitted runs with no finite predicted failure need not have a strict failure-length ordering.
- The existing trials identify candidate mechanisms; broader relevance and novelty still need assessment.

## Remaining questions and navigation

The next useful step is a targeted intervention on a proposed mechanism, such as the attention/head transition, rather than automatically expanding the length sweep. Such interventions are candidates, not approved launch specifications. Model extensions can later test whether an established explanation survives relaxed assumptions.

- [PLAN.md](PLAN.md): research plan and direction-review criteria; consult newer launch specifications for finalized settings.
- [DETECTION_GUIDE.md](DETECTION_GUIDE.md): model and code walkthrough.
- [BASELINE_REPRODUCTION.md](BASELINE_REPRODUCTION.md): historical artifacts and reproduction paths.
- The four result documents above: experiment-specific evidence and caveats.

Keep this overview brief and update it after major experiments. Detailed numbers and new experiment specifications belong in their dedicated documents. This overview is a local working document excluded from Git.
