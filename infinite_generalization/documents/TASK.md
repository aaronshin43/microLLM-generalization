# Task: Stage 4A Non-Binary Target Classification

## Objective

Move beyond binary target existence. Stage 4A keeps the Stage 3 reduced last-query
attention model but changes the output: instead of deciding present vs absent, the model
must output **which target token type is present**, or `n` if no target is present.

This is the first task in **Stage 4 (beyond-binary tasks on the reduced model)**. It is the
smallest step away from Stage 3E: the input structure is unchanged, but the readout becomes
a multi-class label instead of a binary decision.

The central research question is:

**Does the Stage 3 length-generalization result still hold when the model must report target
identity instead of mere presence, and does identity readout introduce any new failure mode?**

## Background

This direction follows the June 19 meeting note (`meetings/June19.md`), which proposed
moving on from binary classification of multiple targets and non-targets. The listed
candidates were non-binary classification, counting, position-dependent tasks, and longer
(sequence) outputs. Stage 4A is the non-binary classification case and the entry point to the
Stage 4 family.

Stage 3 characterized binary existence on the reduced model. Its key findings:

- Constant multiplier fails at long length because target attention dilutes.
- Fixed-log multiplier succeeds when the score margin is large enough.
- Learned-log multiplier succeeds once optimization pushes the worst-case `c * Delta > 1`.

Stage 4A reuses that architecture and asks whether these conclusions survive a richer output.

## Task Definition

Token id convention is unchanged from Stage 3:

```text
target token ids:     0 .. H-1     (H = target_token_count)
non-target token ids: H .. H+M-1   (M = non_target_token_count)
```

Each example:

```text
positive: contains exactly one target token of type h, placed per target_position_mode.
          label = h          (an integer in 0 .. H-1)
negative: all non-target tokens.
          label = n           (class index H, the "no target" class)
```

Output is a single multi-class label over `H + 1` classes:

```text
classes = { 0, 1, ..., H-1, n }
```

Example with target types `r, s, t` (ids 0, 1, 2) and non-target `u`:

```text
"u u s u u"   ->  s
"u u u"       ->  n
"u t u u u"   ->  t
```

The dataset stays balanced: half the examples are positive (target type sampled uniformly
over the `H` types) and half are negative (`n`). Positive examples contain exactly one target
token, as in Stage 3. Multiple simultaneous targets are out of scope for 4A (they belong to
counting and sequence-output tasks).

## Model Changes Versus Stage 3

Keep everything that defines the reduced model and its length-aware behavior:

- query and key projections over token embeddings,
- last-token query attending over all token keys,
- the inverse-temperature multiplier `alpha` with modes `constant`, `log`, `learned_log`,
- softmax attention and the length-aware scaling.

Change only the value pathway and the head:

```text
Stage 3 (binary):
    value_output = [ target_mass, 1 - target_mass ]          # 2-dim
    classifier   = Linear(2, 1)                              # one logit, BCE

Stage 4A (non-binary):
    value_output = [ mass_0, ..., mass_{H-1}, nontarget_mass ]   # (H+1)-dim
    classifier   = Linear(H+1, H+1)                              # H+1 logits, cross-entropy
```

where, using the softmax attention weights over the sequence:

```text
mass_h        = sum of attention on tokens equal to target id h   (for h in 0 .. H-1)
nontarget_mass = sum of attention on all non-target tokens
```

Prediction is `argmax` over the `H + 1` logits; training uses cross-entropy.

This is the exact generalization of Stage 3. With `H = 1` it reduces to
`value_output = [target_mass, nontarget_mass]` with two classes `{ target_0, n }`, which is
the binary present/absent task.

## Hypothesis And Why This Is The Right First Step

- Identity is carried by the attention distribution. Once attention concentrates on the
  target token, the corresponding target slot dominates the value output, so reading identity
  is essentially free.
- Therefore the length-generalization behavior should **inherit** the Stage 3 result:
  constant fails at long length, fixed-log succeeds, and learned-log succeeds when the
  worst-case `c * Delta > 1`.
- The expected failure mode at long length is that target mass dilutes toward zero, so a
  positive example collapses to the `n` class. This is the multi-class analogue of the binary
  false-negative.
- The new thing to check is whether **target-type confusion** ever appears, that is, a
  positive of type `h` predicted as a different target type `h'` rather than as `n`. With
  one-hot identity values and a single target per example this should not happen, so observing
  it would be informative.

Because the input distribution and the length-aware mechanism are unchanged, Stage 4A is a
low-risk bridge that also builds the multi-class infrastructure needed for later Stage 4 tasks.

## Reuse From Stage 3

Stage 4A should reuse, not reimplement, the Stage 3 infrastructure:

- dataset generation with `target_token_count`, `non_target_token_count`, and
  `target_position_mode`,
- chunked evaluation (`test_examples`, `eval_chunk_examples`) so length 10M stays feasible,
- stratified evaluation (`eval_sampling_mode = stratified`) with positives stratified over
  target token id, as in Stage 3E,
- the `alpha` modes and the `c * Delta` diagnostic.

Shared helpers can be imported from `stage3_simplified_attention.py` or factored into a small
common module. The only data change is the label: an integer class index instead of a binary
target flag.

## Diagnostics And Metrics

Per evaluation length, record:

```text
overall accuracy
n-class accuracy            (recall on negatives)
per-target-type accuracy    (recall for each target id h)
mean target attention       (attention mass on the target token, positives)
mean and worst margin Delta  (target vs non-target score margin)
c, c * Delta                (learned_log only; worst case over target types)
```

Also record a small confusion summary for positives:

```text
fraction of positives predicted as the correct target type
fraction predicted as n            (dilution failure)
fraction predicted as another target type   (identity confusion)
```

The asymptotic diagnostic remains the worst-case `c * Delta > 1` over target types, exactly as
in Stage 3E.

## Implementation Plan

### Step 1: New Module

Create `src/stage4a_nonbinary_classification.py` that reuses Stage 3 dataset and evaluation
helpers. Keep the Stage 3 binary code unchanged as a reference.

### Step 2: Multi-Class Model

Add a model (for example `SimplifiedLastQueryAttentionMultiClass`) with the `(H+1)`-dim value
output and an `Linear(H+1, H+1)` head. Reuse the Stage 3 query/key projections and the `alpha`
modes verbatim.

### Step 3: Labels And Loss

Assign positive labels to the target type `h` and negative labels to class `H` (`n`). Train
with cross-entropy. Verify that `H = 1` reproduces the binary task behavior.

### Step 4: Evaluation And Metrics

Adapt the chunked and stratified evaluation to multi-class metrics listed above. Stratify
positive examples over target token id. Write per-length metrics and a per-target-type CSV.

### Step 5: Tests

Add tests under `tests/`:

1. Value output has dimension `H + 1` and the masses sum to 1 per example.
2. `H = 1` reduces to the binary present/absent behavior.
3. Labels are assigned correctly (positive to `h`, negative to `H`).
4. Stratified evaluation balances positive target token ids.
5. Single-chunk random evaluation matches the unchunked dataset.
6. Default config runs end to end on a tiny model.

Use small lengths and example counts so tests stay fast.

### Step 6: Smoke Test

Run a tiny smoke configuration for each `alpha` mode to confirm the pipeline, writing under
`runs/stage4a_nonbinary/`.

### Step 7: Main Runs

Run representative conditions at lengths up to 10M with stratified evaluation:

```text
alpha_mode in { constant, log, learned_log }
target_token_count = 3      (three target types plus the n class)
non_target_token_count = 1
eval_sampling_mode = stratified
```

### Step 8: Analysis Versus Stage 3

Compare against the Stage 3 binary results and answer:

```text
Do the constant / fixed-log / learned-log conclusions still hold for identity output?
Is the dominant long-length failure a collapse to n, as predicted?
Does any target-type confusion appear, separate from the n-collapse?
```

## Done Criteria

This task is complete when:

1. A multi-class reduced model outputs target identity or `n`.
2. Training, chunked evaluation, and stratified evaluation work for the multi-class setting.
3. With `H = 1` the setup reduces to the Stage 3 binary task.
4. Unit tests cover the value output, label assignment, and evaluation.
5. Smoke tests pass for all three `alpha` modes.
6. Representative runs complete under `runs/stage4a_nonbinary/`.
7. A short analysis states whether the Stage 3 conclusions transfer and characterizes the
   long-length failure mode.

## Out Of Scope (Future Stage 4 Tasks)

- Stage 4B: counting the number of targets present.
- Stage 4C: position-dependent tasks (for example output order, or `s` before `t`), which
  require introducing positional information.
- Stage 4D: longer variable-length outputs (sequence output), which require a decoder.
