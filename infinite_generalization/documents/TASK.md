# First Task Definition

## Task Name

**Token-Presence Detection**

## Core Question

Given a sequence of discrete tokens, predict whether a fixed target token appears **anywhere** in the sequence.

Formally, for input sequence:

```text
x = (x_1, x_2, ..., x_L)
```

predict:

```text
y = 1 if there exists i such that x_i = T
y = 0 otherwise
```

where `T` is a fixed target token chosen once for the experiment.

## Why This Task

This is the cleanest starting task for infinite-length generalization because:

- it does not require positional information
- there is a simple length-invariant algorithm
- it is easy to generate unlimited synthetic data
- failures are likely to reflect architectural or optimization issues, not task difficulty

The natural algorithm is:

```text
present = max_i indicator(x_i == T)
```

## Vocabulary

Use a small discrete vocabulary:

- vocabulary size: `V = 16`
- token IDs: `0..15`
- fixed target token: `T = 1`

This can be changed later, but the first experiment should keep it small.

## Sequence Lengths

Training:

- train only on sequences of length `10`

Evaluation:

- `10, 20, 50, 100, 200, 500, 1000`

## Label Distribution

Use a balanced binary dataset:

- 50% positive examples
- 50% negative examples

### Negative examples

- sample all tokens uniformly from the non-target vocabulary `0..15` excluding `T`

### Positive examples

For the initial training set:

- require at least one occurrence of `T`
- force exactly one occurrence of `T`
- sample all remaining tokens from the non-target vocabulary `0..15` excluding `T`

Reason:

- exactly-one-target positives make the signal clean
- they are the most informative case for testing dilution failures
- they are the hardest positive case for existential detection, because the target signal becomes sparse as length grows

Multiple-target positives should not be mixed into the initial training distribution. They should be added as a separate evaluation slice after the exactly-one setup is working, and later as a controlled training-distribution ablation.

## Data Generation Rules

For a sequence of length `L`:

### Negative

1. sample `L` tokens uniformly from `V \\ {T}`
2. assign label `0`

### Positive

1. sample `L - 1` tokens uniformly from `V \\ {T}`
2. choose one index uniformly from `0..L-1`
3. insert `T` at that index
4. assign label `1`

This guarantees that positives contain exactly one target token and negatives contain none.

## Train / Validation / Test Splits

Initial recommendation:

- training set: `50,000` examples at length `10`
- validation set: `10,000` examples at length `10`
- test sets: `10,000` examples per evaluation length

For longer lengths, create separate test sets rather than mixing lengths together.

## Evaluation Slices

For the first report, compute:

1. overall accuracy by sequence length
2. positive-class accuracy by sequence length
3. negative-class accuracy by sequence length
4. exactly-one-target positive accuracy by sequence length

After the first baseline works, add these secondary diagnostic slices:

1. multiple target tokens
2. target near beginning
3. target near middle
4. target near end

The primary long-length test should remain exactly-one positives plus zero-target negatives. Multiple-target positives are useful for confirming robustness, but they are easier than exactly-one positives and can hide length-dependent failures if used as the main evaluation.

## Baseline Model Requirement

The first model should be a non-transformer permutation-invariant baseline:

```text
Embedding
per-token MLP
max pooling across sequence
binary classifier
```

This model should not use:

- positional encoding
- sequence flattening
- recurrent state

## Transformer Model Requirement

The first transformer should also avoid length-specific components:

- no learned absolute positional embeddings
- no flattening of the sequence dimension
- dynamic handling of sequence length

Recommended first transformer:

- `d_model = 64`
- `num_layers = 1`
- `num_heads = 1` or `2`
- max pooling over token representations

## Success Criterion For The First Experiment

The first experiment is successful if:

1. the baseline reaches near-perfect accuracy on train length `10`
2. the baseline retains near-perfect accuracy across much longer lengths
3. the setup is stable enough to use as the reference point for the first transformer experiment

The first experiment is not trying to prove something deep yet. It is trying to establish a clean, trustworthy testbed.
