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
