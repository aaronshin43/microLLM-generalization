# Task: Stage 4B Ablation 3B - Warm-Started Top-K Restricted Softmax

## Objective

The cold-start `topk_softmax_mass` ablation failed before it could test the intended bounded
denominator idea. Across seeds `42`, `43`, and `44`, the hard top-k subset selected only
non-target tokens, so the classifier saw a constant readout and the score-ranking side received
little useful signal to move target tokens into the selected subset.

This task tests the most direct follow-up:

```text
Warm-start top-k from a differentiable full-softmax model.
```

The central question is:

**If the model first learns a target-over-non-target ranking through full softmax, can the
same reduced model then use `topk_softmax_mass` to preserve count information across length?**

This separates two hypotheses:

```text
cold-start ranking failure:
    hard top-k is usable after the ranking is learned, but cannot learn the ranking from
    scratch because target tokens are initially outside the selected subset

top-k readout failure:
    even with a good ranking, the restricted top-k mass does not separate exact count classes
    robustly across length
```

## Background

The cold-start top-k ablation selects the top-$R$ positions by corrected score:

```math
r_j = \alpha(n) s_j,
```

where $s_j$ is the raw query-key score and $\alpha(n)$ is the configured length multiplier.
It then computes a softmax only over the selected set:

```math
J_R(x) =
\operatorname{TopR}\{r_0,r_1,\dots,r_{n-1}\}.
```

If all true target positions enter $J_R(x)$ and true count $k \le R$, the idealized target mass
is:

```math
m_k^{\text{top-}R}
=
\frac{k e^{\alpha\Delta}}{k e^{\alpha\Delta} + (R-k)}.
```

The denominator contains $(R-k)$ instead of $(n-k)$, so it should not dilute with sequence
length. However, this reasoning assumes the ranking condition is already satisfied. The
cold-start runs violated that condition: target positions never entered top-k.

Full softmax is differentiable over all positions:

```math
a_j =
\frac{e^{r_j}}{\sum_\ell e^{r_\ell}}.
```

Even if target scores are initially lower than non-target scores, target positions still
receive nonzero attention mass and therefore receive gradient signal from the count loss.
Warm-start uses this differentiable path to first train the score ranking, then switches to
the hard top-k readout.

## Scope

Implement only the warm-started top-k diagnostic.

Keep unchanged:

- Stage 4B dataset and count labels,
- `constant`, `log`, and `learned_log` alpha modes,
- `topk_softmax_mass` readout definition,
- `top_k = 3` primary setting,
- chunked and stratified evaluation,
- existing cold-start top-k runs for comparison.

Do not implement yet:

- differentiable top-k relaxation,
- ranking auxiliary loss,
- learned or adaptive `top_k`,
- multi-length training,
- multi-target-type counting,
- full-transformer Stage 1/2 changes.

## Warm-Start Design

Use two phases.

### Phase 1: Differentiable Ranking Pretraining

Train or reuse a `softmax_mass` Stage 4B checkpoint with the same base setting:

```text
target_token_count     = 1
non_target_token_count = 1
max_target_count       = 3
target_position_mode   = fixed_start
train_length           = 10
eval_sampling_mode     = stratified
```

The existing e500 baseline checkpoints under `runs/stage4b/` are acceptable warm-start
sources:

```text
runs/stage4b/constant_e500_t1_nt1_k3/model.pt
runs/stage4b/log_e500_t1_nt1_k3/model.pt
runs/stage4b/learned_log_e500_t1_nt1_k3/model.pt
```

The purpose of Phase 1 is not long-length counting success. It is to obtain score-side weights
where target scores rank above non-target scores at the training length.

### Phase 2: Hard Top-K Fine-Tuning

Create a `topk_softmax_mass` model and initialize its score-side parameters from the Phase 1
checkpoint:

```text
query_projection
key_projection
alpha_log_scale_unconstrained
```

Prefer reinitializing the count classifier for the primary diagnostic. The top-k readout has a
different calibration from full-softmax mass, so a fresh classifier makes the question cleaner:

```text
given a learned ranking, can top-k mass learn count thresholds?
```

If classifier-copying is easy, it can be an optional sanity check, but do not make it the
primary result.

Fine-tune with:

```text
readout_mode = topk_softmax_mass
top_k        = 3
```

## Required Implementation

Add a way to warm-start Stage 4B training from an existing checkpoint. A suggested CLI/API is:

```text
--warm-start-checkpoint PATH
--warm-start-mode score_only
```

Supported mode for this task:

```text
score_only:
    load query projection, key projection, and learned alpha parameter
    do not load classifier weights
```

Validation requirements:

- checkpoint `target_token_count`, `non_target_token_count`, `max_target_count`, `d_head`, and
  `alpha_mode` must match the new run config,
- fail with a clear error on mismatches,
- save warm-start metadata into `config.json` and `model.pt`,
- preserve reproducibility for non-warm-start runs.

## Diagnostics

Keep all current Stage 4B and top-k diagnostics, especially:

```text
mean_topk_target_count
mean_topk_target_recall
mean_topk_all_targets_included
mean_topk_non_target_count
mean_topk_target_mass
mean_topk_non_target_mass
mean_target_attention_mass
mean_min_margin_delta
worst_min_margin_delta
readout_finite_fraction
```

For warm-start runs, also record:

```text
warm_start_checkpoint
warm_start_mode
```

Key diagnostic interpretation:

```text
If top-k target recall is high but accuracy fails:
    readout/calibration failure

If top-k target recall remains zero:
    warm start did not solve ranking

If train length succeeds and 10M succeeds:
    cold-start ranking was the main blocker

If train length succeeds but 10M fails:
    top-k may learn finite count thresholds but still lacks length-stable calibration
```

## Implementation Plan

### Step 1: Add Warm-Start Loading

Add checkpoint loading support to `stage4b_counting.py` for score-only initialization.

### Step 2: Thread Metadata

Thread `warm_start_checkpoint` and `warm_start_mode` through:

```text
config dataclass
CLI
saved config JSON
model checkpoint metadata
metrics CSVs
count metrics
confusion outputs
```

### Step 3: Tests

Add or update tests under `infinite_generalization/tests/`:

1. Existing Stage 4B tests still pass.
2. Score-only warm start copies query/key/alpha parameters.
3. Score-only warm start does not copy classifier parameters.
4. Mismatched checkpoint metadata raises a clear error.
5. Chunked evaluation still works for warm-started `topk_softmax_mass`.
6. A tiny warm-start training/eval run completes end to end.

### Step 4: Smoke Runs

Run tiny smoke configurations for:

```text
constant
log
learned_log
```

with:

```text
readout_mode          = topk_softmax_mass
top_k                 = 3
warm_start_mode       = score_only
warm_start_checkpoint = matching softmax_mass smoke checkpoint
```

Use a separate output directory:

```text
runs/stage4b/ablation3_topk_warmstart_smoke/
```

### Step 5: Main Runs

Run the base Stage 4B setting with `top_k = 3` and score-only warm starts from the matching
e500 `softmax_mass` checkpoints.

Output directory:

```text
runs/stage4b/ablation3_topk_warmstart/
```

Recommended run names:

```text
constant_e500_topk3_score_warmstart_t1_nt1_k3
log_e500_topk3_score_warmstart_t1_nt1_k3
learned_log_e500_topk3_score_warmstart_t1_nt1_k3
```

Use the same evaluation setup as the cold-start top-k main runs:

```text
test_examples        = 720
eval_chunk_examples  = 36
eval_lengths         = 10 ... 10000000
max_train_steps      = 16000
```

If seed 42 changes the conclusion, repeat with seeds `43` and `44` without overwriting the
primary run:

```text
runs/stage4b/ablation3_topk_warmstart/seed43/
runs/stage4b/ablation3_topk_warmstart/seed44/
```

### Step 6: Report Update

Update `STAGE4B_COUNTING_TARGET_OCCURRENCES.md` with a short warm-start interpretation.

## Analysis Questions

After the runs complete, answer:

```text
Does score-only warm start make target positions enter the top-k subset?
Does topk_softmax_mass fit the training length after warm start?
Does it avoid collapse at 10M?
If it fails, is the failure now ranking failure or readout/calibration failure?
Does constant scaling benefit most from warm-started top-k?
Do log and learned_log saturate positive counts once the denominator is fixed?
Does this support the hypothesis that cold-start hard selection caused the original failure?
```

## Done Criteria

This task is complete when:

1. Warm-start checkpoint loading is implemented for Stage 4B score-side parameters.
2. Warm-start metadata is saved in configs, checkpoints, and CSV outputs.
3. Tests cover score-only loading and mismatch handling.
4. Smoke runs pass for all three alpha modes.
5. Main warm-start top-k runs complete under
   `runs/stage4b/ablation3_topk_warmstart/`.
6. The Stage 4B report is updated with the warm-start diagnostic result.

## Out Of Scope

- Treating warm-started top-k as the final architecture.
- Differentiable top-k relaxation.
- Ranking auxiliary loss.
- Learned or adaptive `top_k`.
- Multi-length training.
- Multi-target-type counting.
- Full-transformer Stage 1/2 changes.

<!-- Previous completed Ablation 3 task retained below for reference only.

# Task: Stage 4B Ablation 3 - Top-K Restricted Softmax

## Objective

Stage 4B showed that the strict normalized softmax-attention counting baseline can fit the
training length but fails to length-generalize exact counts. The full softmax denominator grows
with the number of non-target positions, so the target count signal becomes length-dependent.

Ablation 1 removed the softmax denominator entirely, but the unnormalized non-target numerator
became a length-growing background feature. Ablation 2 removed the non-target numerator and
showed that a constant target numerator can preserve count, but the diagnostic is not a natural
attention readout and length-aware multipliers can still introduce target-scale drift.

This task tests the next more attention-like intervention:

```text
Ablation 3: select top-k positions by corrected score, then compute a softmax only over that
selected subset.
```

The central question is:

**If the attention denominator is restricted to a fixed-size top-k subset, can the reduced
Stage 4B model preserve count information across length without using an unnormalized readout?**

This should be implemented as a corrected-score top-k subset softmax, not as a post-hoc
full-softmax truncation. The two are mathematically equivalent when the selected set is the
same, but selecting from corrected scores is cleaner and avoids making the full-sequence
softmax denominator part of the readout definition.

## Background

The strict Stage 4B baseline uses full softmax target mass. For one target type, true count
$k$, target score $a$, non-target score $b$, and margin $\Delta=a-b$:

```math
m_k(n)
=
\frac{k e^{\alpha a}}{k e^{\alpha a} + (n-k)e^{\alpha b}}
=
\frac{k e^{\alpha\Delta}}{k e^{\alpha\Delta} + (n-k)}.
```

The failure comes from the $(n-k)$ term. The count $k$ is bounded by `max_target_count`, but
$n$ grows to 10M.

For top-k restricted softmax, use $R$ for the fixed top-k size to avoid confusing it with the
maximum count class $K$. Let the corrected score be:

```math
r_j = \alpha(n) s_j.
```

Select the top-$R$ index set by corrected score:

```math
J_R(x)
=
\operatorname{TopR}\{r_0,r_1,\dots,r_{n-1}\}.
```

Then compute attention only inside that subset:

```math
\tilde a_j
=
\frac{e^{r_j}}{\sum_{\ell\in J_R(x)} e^{r_\ell}}
\quad\text{for }j\in J_R(x),
\qquad
\tilde a_j = 0 \quad\text{otherwise}.
```

The value readout uses this restricted attention distribution:

```math
z =
\sum_{j\in J_R(x)} \tilde a_j v_j.
```

In the idealized Stage 4B setting, if $k \le R$ and all target positions rank above the
non-target positions needed to fill the top-$R$ set, then the target mass becomes:

```math
m_k^{\text{top-}R}
=
\frac{k e^{\alpha a}}{k e^{\alpha a} + (R-k)e^{\alpha b}}
=
\frac{k e^{\alpha\Delta}}{k e^{\alpha\Delta} + (R-k)}.
```

The key change is:

```text
full softmax:       denominator has (n-k)
top-k softmax:      denominator has (R-k)
```

Because $R$ is fixed, the denominator no longer grows with sequence length. However, this does
not guarantee exact counting. The top-k mechanism can fail in two different ways:

```text
ranking failure:
    target tokens do not enter the top-k subset

readout failure:
    target tokens enter the top-k subset, but the restricted mass saturates or otherwise
    fails to separate count classes
```

This ablation should therefore record explicit top-k selection diagnostics, not only final
count accuracy.

## Scope

Implement and run only the corrected-score top-k restricted softmax readout.

Do not implement these variants yet:

- full-softmax truncation as the primary implementation,
- learned or adaptive `top_k`,
- denominator or log-denominator features,
- parallel detector plus sum pooling,
- multi-length training,
- multi-target-type counting.

## Baselines To Compare Against

Strict Stage 4B baseline:

```text
documents/STAGE4B_COUNTING_TARGET_OCCURRENCES.md
runs/stage4b/
```

Ablation 1:

```text
runs/stage4b/ablation1_unnormalized/
```

Ablation 2:

```text
runs/stage4b/ablation2_target_only/
```

Key comparison:

```text
strict softmax_mass baseline:
    denominator includes all n positions
    positive counts collapse to count 0 at 10M

ablation1 unnormalized_sum:
    target numerator preserves count
    non-target numerator grows with length
    10M accuracy = 0.250

ablation2 target_numerator_only:
    constant target numerator succeeds
    log and learned_log overestimate counts because target scale drifts upward

ablation3 topk_softmax_mass:
    denominator includes only fixed top_k positions
    test whether this preserves a bounded, attention-like count signal
```

## Model Change

Keep the Stage 4B dataset, labels, loss, multiplier modes, and chunked evaluation unchanged.

Add another readout mode, for example:

```text
readout_mode = softmax_mass | unnormalized_sum | target_numerator_only | topk_softmax_mass
```

Add a config/CLI field for the restricted attention size:

```text
top_k = 3
```

Use the name `top_k` in configs and CSVs, but use $R$ in document math when needed to avoid
confusion with `max_target_count`.

The existing modes must remain reproducible:

```text
softmax_mass:
    value_output = [full-softmax target mass(es), full-softmax non-target mass]

unnormalized_sum:
    value_output = [target numerator sum(s), non-target numerator sum]

target_numerator_only:
    value_output = [target numerator sum]
```

Ablation 3 should use:

```text
topk_softmax_mass:
    select top_k indices by corrected score
    compute softmax only over those selected corrected scores
    value_output = [top-k target mass(es), top-k non-target mass]
```

For the current base setting $H = 1$:

```math
m_t^{\text{top-}R}
=
\sum_{j\in J_R(x):\,x_j=t}
\frac{e^{r_j}}{\sum_{\ell\in J_R(x)} e^{r_\ell}},
\qquad
m_{\text{non}}^{\text{top-}R}
=
1 - m_t^{\text{top-}R}.
```

The classifier remains:

```text
classifier = Linear(H + 1, K + 1)
loss       = cross-entropy over count classes
```

For now, this ablation only needs to support the base Stage 4B setting:

```text
target_token_count     = 1
non_target_token_count = 1
max_target_count       = 3
```

The main run should use `top_k = 3`, matching `max_target_count`. If the primary result is
ambiguous, add sensitivity checks with `top_k = 4` and `top_k = 6` without overwriting the
primary runs.

## Numerical Handling

The top-k readout should be computed from corrected scores:

```text
corrected_scores = alpha(length) * raw_scores
top_values, top_indices = topk(corrected_scores, k=top_k)
top_weights = softmax(top_values)
```

Do not compute the top-k readout by first applying full softmax over all $n$ positions and
then truncating the resulting weights. That form is mathematically equivalent after
renormalization in exact arithmetic, but it obscures the intended denominator and can add
unnecessary numerical and memory pressure.

Use a numerically stable subset softmax. Keep existing finite-value diagnostics:

```text
mean_max_corrected_score
max_corrected_score
mean_max_readout_value
max_readout_value
readout_finite_fraction
```

If `top_k > length`, use `min(top_k, length)` and record the effective value, or raise a clear
error. For the planned runs, `top_k <= train_length` and this should not occur.

## Diagnostics

Keep the existing Stage 4B metrics and add/retain fields that make the top-k behavior clear:

```text
readout_mode
top_k
effective_top_k
overall accuracy
per-count recall
count confusion matrix
mean predicted count
mean absolute count error
mean top-k target mass by true count
mean top-k non-target mass by true count
mean full-softmax target attention mass by true count, if still available
mean/worst target-vs-non-target margin Delta
c and c * Delta for learned_log
finite readout fraction
```

Add top-k selection diagnostics:

```text
mean_topk_target_count
mean_topk_target_recall
mean_topk_all_targets_included
mean_topk_non_target_count
```

Definitions:

```text
topk_target_count:
    number of true target positions selected into the top-k set

topk_target_recall:
    topk_target_count / true_count for true_count > 0
    leave empty, NaN, or define separately for true_count = 0

topk_all_targets_included:
    1 if topk_target_count == true_count, else 0

topk_non_target_count:
    number of selected non-target positions
```

These diagnostics are necessary to distinguish ranking failure from readout failure.

## Implementation Plan

### Step 1: Add Top-K Softmax Readout Mode

Update the Stage 4B counting model to support:

```text
topk_softmax_mass
```

Keep `softmax_mass` as the default. Keep `unnormalized_sum` and `target_numerator_only`
working.

### Step 2: Add `top_k`

Thread `top_k` through:

```text
config dataclass
CLI
saved config JSON
model checkpoint metadata
metrics CSVs
count metrics
confusion outputs
```

The value should be ignored by non-top-k modes except for being saved as metadata if that is
simpler.

### Step 3: Implement Subset Softmax

Implement the readout from corrected scores:

```text
1. Select top_k positions by corrected score.
2. Apply softmax to only the selected corrected scores.
3. Aggregate selected weights by token type.
4. Feed [top-k target mass(es), top-k non-target mass] to the classifier.
```

For ties, use the deterministic behavior of the underlying `topk` implementation. The report
should mention that ties are not expected to matter in the learned-score regime.

### Step 4: Evaluation And CSVs

Add the top-k diagnostics listed above. Make sure chunked evaluation still matches
single-chunk evaluation.

The existing full-softmax diagnostics may remain for comparison, but the classifier readout
must be the top-k restricted softmax mass.

### Step 5: Tests

Add or update tests under `infinite_generalization/tests/`:

1. Existing `softmax_mass`, `unnormalized_sum`, and `target_numerator_only` tests still pass.
2. `topk_softmax_mass` output has shape `(batch, H + 1)`.
3. The top-k readout is computed from corrected scores and assigns zero mass to positions
   outside the selected subset.
4. For a small hand-constructed example, subset-softmax mass matches renormalized full-softmax
   mass over the same selected top-k indices.
5. With `top_k = max_target_count` and targets ranked highest, all target positions are
   included for counts up to `max_target_count`.
6. Top-k diagnostics report ranking success and ranking failure correctly.
7. Chunked evaluation matches single-chunk evaluation for `topk_softmax_mass`.
8. A tiny config runs end to end for `topk_softmax_mass`.

### Step 6: Smoke Runs

Run tiny smoke configurations for:

```text
constant
log
learned_log
```

with:

```text
readout_mode = topk_softmax_mass
top_k = 3
```

Use a separate smoke output directory:

```text
runs/stage4b/ablation3_topk_smoke/
```

### Step 7: Main Runs

Run the same base Stage 4B setting:

```text
target_token_count     = 1
non_target_token_count = 1
max_target_count       = 3
target_position_mode   = fixed_start
train_length           = 10
eval_sampling_mode     = stratified
eval_lengths           = 10 ... 10000000
top_k                  = 3
```

Use a separate output directory:

```text
runs/stage4b/ablation3_topk/
```

Recommended run names:

```text
constant_e500_topk3_t1_nt1_k3
log_e500_topk3_t1_nt1_k3
learned_log_e500_topk3_t1_nt1_k3
```

If e500 is undertrained, add longer runs without overwriting e500.

## Analysis Questions

After the runs complete, answer:

```text
Does topk_softmax_mass fit the training length?
Does it avoid count-0 collapse at 10M?
Are all true target positions selected into the top-k subset at long length?
If a run fails, is it a ranking failure or a readout failure?
Does constant scaling preserve count-sensitive restricted mass?
Do log and learned_log saturate positive counts once the denominator is fixed?
Is top_k = 3 enough, or do top_k = 4 or top_k = 6 sensitivity checks change the result?
How should this diagnostic be interpreted relative to the strict full-softmax baseline?
```

## Done Criteria

This task is complete when:

1. `readout_mode=topk_softmax_mass` is implemented without breaking existing readout modes.
2. `top_k` is threaded through config, CLI, checkpoints, and CSV outputs.
3. Tests cover the new readout path and top-k diagnostics.
4. Smoke runs pass for all three alpha modes.
5. Main top-k ablation runs complete under `runs/stage4b/ablation3_topk/`.
6. The Stage 4B report is updated with a clearly labeled diagnostic interpretation.

## Out Of Scope

- Treating top-k restricted softmax as the final proposed architecture.
- Learned or adaptive top-k selection.
- Multi-length training.
- Multi-target-type counting.
- Full-transformer Stage 1/2 changes.

-->

<!-- Previous completed Ablation 2 task retained below for reference only.

# Task: Stage 4B Ablation 2 - Target Numerator Only

## Objective

Stage 4B Ablation 1 replaced normalized softmax attention mass with the full unnormalized
readout:

```text
[target numerator sum, non-target numerator sum]
```

It showed that the target numerator preserves count information, but the non-target numerator
introduces a length-growing background scale. At length 10M, the classifier still collapsed to
predicting count 0 for every example.

This task runs the next diagnostic ablation:

```text
Ablation 2: expose only the target numerator sum to the count classifier.
```

The central question is:

**If the length-growing non-target background is removed, is the target numerator alone a
stable count signal?**

This is a diagnostic upper-bound style ablation. It is not intended to be a natural
replacement architecture for the Stage 3/4A reduced model. Its purpose is to isolate whether
the target numerator itself contains length-stable count information.

## Background

The strict Stage 4B baseline uses normalized softmax mass:

```math
m_k(n) =
\frac{k e^{\alpha\Delta}}{k e^{\alpha\Delta} + (n-k)}.
```

This failed because the count signal is represented only as a length-dependent relative mass.

Ablation 1 used unnormalized numerator sums:

```math
u_t =
\sum_{j:\,x_j=t} e^{\alpha s_j},
\qquad
u_{\text{non}} =
\sum_{j:\,x_j\ \text{non-target}} e^{\alpha s_j}.
```

In the constant run, $u_t$ preserved the count factor exactly across length:

```text
count 1: 57.90 at length 10 and 57.90 at length 10M
count 2: 115.80 at length 10 and 115.80 at length 10M
count 3: 173.71 at length 10 and 173.71 at length 10M
```

But $u_{\text{non}}$ grew from about `138` at length 10 to about `1.38e8` at length 10M, and
the learned classifier used that non-target feature strongly. The result was still count-0
collapse.

Ablation 2 removes $u_{\text{non}}$ from the classifier input and tests whether $u_t$ alone is
sufficient.

## Scope

Implement and run only the target-numerator-only diagnostic.

Do not implement these variants yet:

- denominator or log-denominator readout,
- normalized non-target readout,
- parallel detector plus sum pooling,
- multi-length training,
- multi-target-type counting.

## Baselines To Compare Against

Strict Stage 4B baseline:

```text
documents/STAGE4B_COUNTING_TARGET_OCCURRENCES.md
runs/stage4b/
```

Ablation 1:

```text
runs/stage4b/ablation1_unnormalized/
```

Key comparison:

```text
strict softmax_mass baseline:
    target count signal becomes length-dependent normalized mass
    10M accuracy = 0.250

ablation1 unnormalized_sum:
    target numerator preserves count
    non-target numerator grows with length
    10M accuracy = 0.250

ablation2 target_numerator_only:
    test whether target numerator alone avoids both failure modes
```

## Model Change

Keep the Stage 4B dataset, labels, loss, multiplier modes, and chunked evaluation unchanged.

Add another readout mode, for example:

```text
readout_mode = softmax_mass | unnormalized_sum | target_numerator_only
```

The existing modes must remain reproducible:

```text
softmax_mass:
    value_output = [normalized target mass(es), normalized non-target mass]

unnormalized_sum:
    value_output = [target numerator sum(s), non-target numerator sum]
```

Ablation 2 should use:

```text
target_numerator_only:
    value_output = [target numerator sum]
```

For the current base setting $H = 1$:

```math
u_t =
\sum_{j:\,x_j=t} e^{\alpha s_j}.
```

The classifier becomes:

```text
classifier = Linear(1, K + 1)
loss       = cross-entropy over count classes
```

If the implementation is easier with a fixed two-dimensional input, using
`[target numerator sum, 0]` is acceptable, but the report must state that the second dimension
is a constant dummy feature. Prefer `Linear(1, K + 1)` if it can be done cleanly.

For now, this ablation only needs to support `target_token_count = 1`. If supporting general
$H$ is straightforward, the natural extension is:

```text
value_output = [sum over all target-type numerator sums]
```

because the Stage 4B label is total target occurrence count, not target identity.

## Numerical Handling

Use the same numerator definition as Ablation 1:

```math
e^{\alpha s_j}.
```

Keep existing finite-value diagnostics:

```text
mean_max_corrected_score
max_corrected_score
mean_max_readout_value
max_readout_value
readout_finite_fraction
```

If overflow appears, do not silently clip. Use a documented stable variant and record it in
the config/report.

## Diagnostics

Keep the existing Stage 4B metrics and add/retain fields that make the readout clear:

```text
readout_mode
overall accuracy
per-count recall
count confusion matrix
mean predicted count
mean absolute count error
mean target readout by true count
mean normalized target attention mass by true count
mean/worst target-vs-non-target margin Delta
c and c * Delta for learned_log
finite readout fraction
```

For this ablation, `mean_non_target_readout` should be omitted, left empty, or explicitly set
to a dummy value only if the implementation needs a dummy input. Do not label a diagnostic
normalized non-target attention mass as a classifier readout.

## Implementation Plan

### Step 1: Add Target-Only Readout Mode

Update the Stage 4B counting model to support:

```text
target_numerator_only
```

Keep `softmax_mass` as the default. Keep `unnormalized_sum` working.

### Step 2: Classifier Dimension

Make the classifier input dimension depend on `readout_mode`:

```text
softmax_mass:          H + 1
unnormalized_sum:      H + 1
target_numerator_only: 1
```

For this task, it is acceptable to restrict `target_numerator_only` to `target_token_count=1`
and raise a clear error for `H > 1`.

### Step 3: Evaluation And CSVs

Thread `readout_mode` through config, CLI, model checkpoint metadata, metrics CSVs, count
metrics, and confusion outputs.

Make sure chunked evaluation still matches single-chunk evaluation.

### Step 4: Tests

Add or update tests under `infinite_generalization/tests/`:

1. Existing `softmax_mass` and `unnormalized_sum` tests still pass.
2. `target_numerator_only` output has shape `(batch, 1)`.
3. For identical target scores, the target-only readout is proportional to true count.
4. The classifier output dimension remains `K + 1`.
5. Chunked evaluation matches single-chunk evaluation for `target_numerator_only`.
6. A tiny config runs end to end for `target_numerator_only`.

### Step 5: Smoke Runs

Run tiny smoke configurations for:

```text
constant
log
learned_log
```

with:

```text
readout_mode = target_numerator_only
```

### Step 6: Main Runs

Run the same base Stage 4B setting:

```text
target_token_count     = 1
non_target_token_count = 1
max_target_count       = 3
target_position_mode   = fixed_start
train_length           = 10
eval_sampling_mode     = stratified
eval_lengths           = 10 ... 10000000
```

Use a separate output directory:

```text
runs/stage4b/ablation2_target_only/
```

Recommended run names:

```text
constant_e500_t1_nt1_k3
log_e500_t1_nt1_k3
learned_log_e500_t1_nt1_k3
```

If e500 is undertrained, add a longer run without overwriting e500.

## Analysis Questions

After the runs complete, answer:

```text
Does target_numerator_only fit the training length?
Does it avoid count-0 collapse at 10M?
Does constant scaling now extrapolate, since target numerator is length-stable when scores are fixed?
Do log and learned_log introduce target-readout scale drift even without non-target readout?
Is the count signal truly represented by target numerator magnitude?
How should this diagnostic be interpreted relative to the more natural target/non-target readouts?
```

## Done Criteria

This task is complete when:

1. `readout_mode=target_numerator_only` is implemented without breaking `softmax_mass` or
   `unnormalized_sum`.
2. Tests cover the new readout mode.
3. Smoke runs pass for all three alpha modes.
4. Main target-only ablation runs complete under `runs/stage4b/ablation2_target_only/`.
5. The Stage 4B report is updated with a clearly labeled diagnostic interpretation.

## Out Of Scope

- Treating target-only readout as the final proposed architecture.
- Denominator or log-denominator features.
- Normalized non-target readout variants.
- Parallel detector plus sum pooling.
- Multi-length training.
- Multi-target-type counting.

-->

<!-- Previous completed Ablation 1 task retained below for reference only.

# Task: Stage 4B Ablation 1 - Unnormalized Attention Sum

## Objective

Stage 4B showed that the strict normalized softmax-attention counting baseline can fit the
training length but fails to length-generalize exact counts. At long length, all final e500
runs collapse to predicting count 0.

This task starts the follow-up ablation suite, but **only Ablation 1** should be implemented
and run for now:

```text
Ablation 1: replace the normalized softmax attention mass readout with an unnormalized
attention sum readout.
```

The central question is:

**Does Stage 4B counting fail because softmax normalization removes the absolute count scale?**

This ablation should be treated as a diagnostic experiment, not as a replacement for the main
Stage 4B conclusion.

## Background

The current strict baseline reads count information through normalized attention mass:

```math
m_k(n) =
\frac{k e^{\alpha\Delta}}{k e^{\alpha\Delta} + (n-k)}.
```

This is a convex-average statistic. It can distinguish counts at the training length, but it
does not preserve absolute multiplicity in a length-invariant way. Depending on the regime,
positive counts can collapse toward 0 target mass or saturate toward 1 target mass.

Ablation 1 removes the softmax denominator from the value readout. For a target type $h$, use
the unnormalized numerator:

```math
u_h(n) =
\sum_{j:\,x_j=t_h} e^{\alpha s_j}.
```

For the single-target-type base case with $k$ target occurrences and target score $a$:

```math
u_t(n) = k e^{\alpha a}.
```

Unlike $m_k(n)$, this preserves an explicit multiplicative factor $k$. The ablation therefore
tests whether count information becomes more stable once the readout is no longer divided by
the full sequence-level denominator.

## Scope

Implement and run only the unnormalized attention sum ablation.

Do not implement the other candidate ablations yet:

- denominator or log-denominator readout,
- parallel detector plus sum pooling,
- hand-coded count or length features,
- decoder-style outputs.

Those can be considered after Ablation 1 is analyzed.

## Baseline To Compare Against

Use the Stage 4B strict baseline report and run directory as the comparison point:

```text
documents/STAGE4B_COUNTING_TARGET_OCCURRENCES.md
runs/stage4b/
```

The most important baseline runs are the final e500 runs:

```text
runs/stage4b/constant_e500_t1_nt1_k3
runs/stage4b/log_e500_t1_nt1_k3
runs/stage4b/learned_log_e500_t1_nt1_k3
```

The baseline result to compare against is:

```text
train length fit: succeeds
10M evaluation: all examples predicted as count 0
balanced accuracy at 10M: 0.250
```

## Model Change

Keep the Stage 4B dataset, labels, classifier loss, multiplier modes, and chunked evaluation
unchanged.

Add a readout mode flag, for example:

```text
readout_mode = softmax_mass | unnormalized_sum
```

The existing behavior should remain the default:

```text
softmax_mass:
    value_output = [normalized target mass(es), normalized non-target mass]
```

Ablation 1 should use:

```text
unnormalized_sum:
    value_output = [target numerator sum(s), non-target numerator sum]
```

For general $H$ target token types:

```math
u_h =
\sum_{j:\,x_j=t_h} e^{\alpha s_j},
\qquad
u_{\text{non}} =
\sum_{j:\,x_j\ \text{non-target}} e^{\alpha s_j}.
```

The classifier shape stays the same:

```text
classifier = Linear(H + 1, K + 1)
loss       = cross-entropy over count classes
```

For the base setting $H = 1$:

```text
value_output = [target_unnormalized_sum, non_target_unnormalized_sum]
classifier   = Linear(2, K + 1)
```

Important implementation detail: do not remove or break the current normalized baseline path.
The new readout mode should be additive so the original Stage 4B runs remain reproducible.

## Numerical Handling

The unnormalized numerator is based on exponentiated corrected scores:

```math
e^{\alpha s_j}.
```

Implement this carefully enough for the existing 10M evaluation sweep. If numerical overflow
appears, prefer a documented log-space or stable variant over silent clipping. If a stable
variant is used, record it clearly in the config and report because it changes the exact
readout interpretation.

For the first pass, keep the implementation as close as possible to the literal numerator
definition and inspect whether the observed score ranges are safe.

## Diagnostics

Keep the existing Stage 4B metrics and add enough metadata to distinguish the ablation from
the baseline:

```text
readout_mode
overall accuracy
per-count recall
count confusion matrix
mean predicted count
mean absolute count error
mean target readout by true count
mean non-target readout by true count
mean and worst target/non-target margin Delta
c and c * Delta for learned_log
```

If possible, retain the normalized attention mass as a diagnostic even when the classifier
uses `unnormalized_sum`. This makes it easier to compare "what the classifier saw" against
"what the original baseline would have seen."

## Implementation Plan

### Step 1: Add Readout Mode

Update the Stage 4B counting model so it supports both:

```text
softmax_mass
unnormalized_sum
```

The existing default path must remain `softmax_mass`.

### Step 2: Evaluation And CSVs

Thread `readout_mode` through config, CLI, saved config JSON, metrics CSVs, count metrics, and
confusion outputs.

Make sure chunked evaluation still works at length 10M.

### Step 3: Tests

Add or update tests under `infinite_generalization/tests/`:

1. Existing `softmax_mass` tests still pass unchanged.
2. `unnormalized_sum` value output equals the sum of $e^{\alpha s_j}$ over matching token
   positions on a small deterministic batch.
3. For identical target scores, the target readout increases with true count.
4. The count head still has output dimension `K + 1`.
5. Chunked evaluation matches single-chunk evaluation for `unnormalized_sum`.
6. A tiny config runs end to end for `unnormalized_sum`.

### Step 4: Smoke Runs

Run tiny smoke configurations for `unnormalized_sum` with:

```text
constant
log
learned_log
```

Use small lengths, small `K`, and small train steps first.

### Step 5: Main Runs

Run the same base Stage 4B setting used in the strict baseline:

```text
target_token_count     = 1
non_target_token_count = 1
max_target_count       = 3
target_position_mode   = fixed_start
train_length           = 10
eval_sampling_mode     = stratified
eval_lengths           = 10 ... 10000000
```

Use an output directory that clearly separates this ablation from the baseline:

```text
runs/stage4b/ablation1_unnormalized/
```

Recommended run names:

```text
constant_e500_t1_nt1_k3
log_e500_t1_nt1_k3
learned_log_e500_t1_nt1_k3
```

If the first run shows that e500 is clearly undertrained, add a longer run rather than
overwriting the e500 output.

## Analysis Questions

After the runs complete, answer:

```text
Does unnormalized_sum fit the training length?
Does it avoid the count-0 collapse at 10M?
Does constant scaling become enough when the denominator is removed?
Does log or learned_log create a new scale problem because the numerator grows with length?
Are count classes separated by the unnormalized target readout at long length?
Is the original failure mainly caused by softmax normalization, or by another bottleneck?
```

## Done Criteria

This task is complete when:

1. `readout_mode=unnormalized_sum` is implemented without breaking the existing
   `softmax_mass` baseline.
2. Tests cover the new readout path and chunked evaluation.
3. Smoke runs pass for all three alpha modes.
4. Main ablation runs complete under `runs/stage4b/ablation1_unnormalized/`.
5. A short follow-up section or report update compares Ablation 1 against the strict Stage 4B
   baseline.

## Out Of Scope

- Denominator or log-denominator features.
- Parallel detector plus sum pooling.
- Multi-target-type counting.
- Multi-length training.
- Position-dependent tasks.
- Any report rewrite beyond the Ablation 1 comparison needed after the run.

<!-- Previous completed Stage 4B baseline task retained below for reference only.

# Task: Stage 4B Counting Target Occurrences

## Objective

Move beyond target identity classification. Stage 4B keeps the Stage 3/4A reduced
last-query attention model, but changes the output from "which target type is present" to
"how many target tokens are present."

The central research question is:

**Can the reduced length-aware attention model length-generalize on target counting, or does
softmax normalization make absolute multiplicity fundamentally harder than presence and
identity?**

Stage 4A showed that the Stage 3 existence story transfers to non-binary identity
classification: constant fails at long length, fixed-log succeeds, and learned-log succeeds
once the worst-case $c * \Delta$ is large enough. Counting is the next step because it
requires a new statistic: multiplicity rather than presence or identity.

## Background

This direction follows the June 19 meeting note (`meetings/June19.md`), which proposed
moving on from binary classification of multiple targets and non-targets. Stage 4A covered
the first proposed step, non-binary classification. Stage 4B covers the next proposed step:
counting the number of targets present.

The important difference from Stage 4A is that a correct model must distinguish examples
that all contain targets:

```text
"u u t u u"       -> 1
"u t u t u"       -> 2
"t u t u t"       -> 3
"u u u u u"       -> 0
```

This is not just a richer readout of the same existence signal. It asks whether the model can
recover an absolute count from a normalized attention distribution.

## Task Definition

Token id convention is unchanged:

```text
target token ids:     0 .. H-1     (H = target_token_count)
non-target token ids: H .. H+M-1   (M = non_target_token_count)
```

Each example has sequence length `n` and contains exactly `k` target-token occurrences, where:

```text
k in {0, 1, ..., K}     (K = max_target_count)
```

The label is the total number of target occurrences:

```text
label = k
classes = {0, 1, ..., K}
```

For the first Stage 4B experiment, use total target occurrence count, not distinct target type
count. If a target type appears twice, it contributes 2 to the label.

Initial recommended setting:

```text
target_token_count = 1
non_target_token_count = 1
max_target_count = 3 or 5
label = total number of target tokens
```

After the single-target-type setting is understood, extend to:

```text
target_token_count = 3
non_target_token_count = 1 or more
label = total number of target tokens, regardless of type
```

For `k > 0`, sample `k` distinct target positions without replacement according to the target
position mode. For `H > 1`, sample each target occurrence's type uniformly from the `H` target
types unless a later diagnostic needs controlled type composition. Fill all remaining
positions with non-target tokens.

The dataset should be balanced over count classes by default: each class `k in {0, ..., K}`
appears equally often in train and evaluation, subject to divisibility.

## Model Baseline

The first implementation should be a strict reuse of the Stage 4A reduced model family. Keep:

- query and key projections over token embeddings,
- last-token query attending over all token keys,
- the inverse-temperature multiplier `alpha` with modes `constant`, `log`, and `learned_log`,
- softmax attention and chunked length evaluation.

Change only the output classes:

```text
value_output = [mass_0, ..., mass_{H-1}, nontarget_mass]   # H + 1 dimensions
classifier   = Linear(H + 1, K + 1)                        # count logits
loss         = cross-entropy over count classes
```

For `H = 1`, this reduces to:

```text
value_output = [target_mass, nontarget_mass]
classifier   = Linear(2, K + 1)
```

This baseline is intentionally strict. It tests whether the normalized attention masses that
were sufficient for existence and identity are also sufficient for counting.

Do not add a hand-coded length feature, count feature, or decoder in the first baseline. Those
can be considered later only if the strict baseline exposes a clear limitation.

## Theory And Expected Failure Modes

For a simplified single-target-type case with `k` target tokens, one non-target type, and a
target/non-target score margin $\Delta$, the total target attention mass is approximately:

```math
m_k(n) = \frac{k e^{\alpha\Delta}}{k e^{\alpha\Delta} + (n-k)}
```

This differs from Stage 3/4A, where the key question was whether a single target receives
enough attention to avoid collapsing into the non-target class.

Counting has a different tension:

- With constant $\alpha$, target mass dilutes as `n` grows, so all positive counts can
  collapse toward class 0.
- With very strong log scaling where $c\Delta > 1$, target mass approaches 1 for every
  fixed `k > 0`, so the model may preserve presence but lose count information among
  positive classes.
- Near the critical regime $c\Delta \approx 1$, the mass can retain count-dependent
  structure because the limit is roughly count-sensitive rather than fully saturated.

In the idealized case $\alpha = c \log n$, the behavior is:

```math
m_k(n) = \frac{k n^{c\Delta}}{k n^{c\Delta} + (n-k)}
```

So the asymptotic regimes are:

```text
c * Delta < 1: positives collapse toward 0 target mass
c * Delta = 1: target mass remains count-dependent, approximately k / (k + 1)
c * Delta > 1: all positive counts saturate toward target mass 1
```

This means the Stage 3/4A success condition $c * \Delta > 1$ may not be the right success
condition for exact counting. Stage 4B should test whether counting requires a calibrated
margin rather than merely a sufficiently large margin.

Expected failure modes:

```text
positive-to-zero collapse:
    target examples predicted as count 0 because attention dilutes

positive-count collapse:
    counts 1, 2, ..., K become hard to separate because target mass saturates

adjacent-count confusion:
    examples are usually recognized as positive but misclassified as nearby counts

type-composition sensitivity:
    with H > 1, total-count accuracy depends on which target types appear
```

## Reuse From Previous Stages

Stage 4B should reuse the existing Stage 3/4A infrastructure where possible:

- token-id conventions,
- reduced last-query attention model,
- `constant`, `log`, and `learned_log` alpha modes,
- chunked evaluation for very long lengths,
- stratified evaluation,
- margin and $c * \Delta$ diagnostics,
- CSV metric writing and run directory conventions.

The main new infrastructure is dataset generation for multiple target occurrences and
stratification by count class.

## Diagnostics And Metrics

Per evaluation length, record:

```text
overall accuracy
per-count recall for each k in {0, ..., K}
count confusion matrix or compact confusion summary
mean predicted count
mean absolute count error
mean target attention mass by true count
mean non-target attention mass by true count
mean and worst target/non-target margin Delta
c and c * Delta for learned_log
```

For `H > 1`, also record:

```text
per-target-type attention mass
accuracy by target type composition when feasible
worst target-type margin
```

The most important plots or tables should show how count classes separate or collapse as
length increases.

## Implementation Plan

### Step 1: Dataset

Add Stage 4B dataset generation for count labels. Support:

```text
max_target_count
target_token_count
non_target_token_count
target_position_mode
eval_sampling_mode = stratified
```

Stratified evaluation should balance examples over count classes. For `H > 1`, positive
examples should also sample target types uniformly unless a controlled diagnostic is requested.

### Step 2: Model And Loss

Create a Stage 4B model or adapt the Stage 4A model so the value output feeds a
`Linear(H + 1, K + 1)` count head. Use cross-entropy over count classes.

### Step 3: Evaluation

Adapt chunked evaluation so length 10M remains feasible. Evaluation must not materialize huge
full datasets when chunking is enabled.

### Step 4: Tests

Add tests under `infinite_generalization/tests/`:

1. Generated labels equal the true number of target tokens.
2. Count classes are balanced under stratified sampling.
3. Target positions are distinct when `k > 1`.
4. Value masses still sum to 1 per example.
5. The count head has output dimension `K + 1`.
6. Chunked evaluation matches unchunked evaluation on a small deterministic setting.
7. A tiny config runs end to end for each alpha mode.

### Step 5: Smoke Runs

Run tiny smoke configurations for:

```text
constant
log
learned_log
```

Use small lengths, small `K`, and small train steps first.

### Step 6: Main Runs

Run the first representative suite with:

```text
target_token_count = 1
non_target_token_count = 1
max_target_count = 3 or 5
eval_sampling_mode = stratified
alpha_mode in {constant, log, learned_log}
```

Then extend to:

```text
target_token_count = 3
non_target_token_count = 1 or more
max_target_count = 3 or 5
```

### Step 7: Analysis

Answer:

```text
Does constant scaling fail by positive-to-zero collapse?
Does fixed-log or learned-log preserve exact count, or only presence?
Is the Stage 3/4A condition c * Delta > 1 still helpful, or does counting require calibration
near c * Delta = 1?
Which count classes fail first as length increases?
Does adding multiple target types create type-composition sensitivity?
```

## Done Criteria

This task is complete when:

1. Stage 4B can generate examples with `k in {0, ..., K}` target occurrences.
2. The model outputs count classes with cross-entropy loss.
3. Chunked and stratified evaluation work for count labels.
4. Tests cover dataset labels, class balance, value masses, output dimension, and chunked eval.
5. Smoke runs pass for all alpha modes.
6. Representative runs complete under a Stage 4B run directory.
7. A short report explains whether counting follows the Stage 3/4A length-generalization story
   or exposes a new limitation of normalized attention.

## Out Of Scope

- Position-dependent tasks such as "`s` before `t`".
- Longer variable-length sequence outputs.
- Adding a decoder.
- Adding hand-coded count or length features before the strict normalized-attention baseline is
  evaluated.
-->
