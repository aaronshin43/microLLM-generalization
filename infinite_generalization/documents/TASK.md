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
