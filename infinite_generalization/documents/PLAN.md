# Infinite-Length Generalization Plan


## Objective

Study generalization in micro LLMs / tiny sequence models with fewer than 1M parameters, focusing on **infinite-length generalization**:

> If a model is trained on short sequences only, can the exact same learned computation continue to work on much longer sequences?

The initial scope is intentionally narrow:

- synthetic tasks
- tiny models
- no positional information unless explicitly added later
- exhaustive length-sweep evaluation

## Initial Research Question

For a simple existential sequence task with no positional requirement:

> If a model is trained only on length-10 sequences, how well does it generalize to much longer lengths such as 20, 50, 100, 200, 500, and 1000?

The deeper question is not just whether the model can run on longer inputs. It is whether it learns a **length-invariant algorithm** rather than a length-specific shortcut.

## Initial Experimental Strategy

Start with the simplest controlled progression:

1. non-transformer baseline
2. minimal transformer
3. multi-length training intervention
4. pooling ablation
5. only then add architectural complexity

This keeps interpretation clean and makes failures informative.

## Implementation Decision

For the initial phase, use a **fresh lightweight PyTorch implementation**, not `nanoGPT`.

Reason:

- the first task is binary classification, not causal language modeling
- the first models should be tiny encoder-style models, not a full LM stack
- we want complete control over pooling, positional encoding, attention inspection, and length sweeps
- a small custom codebase will be easier to audit and modify

`nanoGPT` can still become useful later if the project shifts toward next-token prediction or more LM-like sequence tasks.

## First Task

The first task is defined in [`TASK.md`](TASK.md).

Short version:

- input: a token sequence
- output: binary label for whether a fixed target token appears anywhere
- train only on length 10
- test on much longer lengths

This task is intentionally simple so that failures can be attributed to architecture or optimization rather than task difficulty.

## Stage Plan

## Stage 0: Non-Transformer Baseline

Model:

- token embedding
- per-token MLP
- max pooling over sequence
- binary classifier

Goal:

- establish a known-good permutation-invariant baseline
- verify that the task itself is easy and length-generalizable

Success criterion:

- near-perfect generalization across all test lengths, including large lengths

## Stage 1: Minimal Transformer

Start with:

- 1 transformer encoder layer
- 1 head or 2 heads
- `d_model = 64`
- no positional encoding
- max pooling

Goal:

- test whether a tiny transformer can discover an equally length-invariant solution

Main questions:

- does it extrapolate beyond length 10?
- does accuracy degrade smoothly or abruptly?
- do attention patterns sharpen onto the target token?

## Stage 2A: Multi-Length Training Intervention

Keep the Stage 1 transformer architecture fixed:

- 1 transformer encoder layer
- 1 head or 2 heads
- `d_model = 64`
- no positional encoding
- max pooling

Change only the training length distribution:

- Stage 1 trains on fixed length `10`
- Stage 2A trains on multiple short lengths, initially `[10, 20, 50, 100]`

Goal:

- test whether the same tiny transformer can learn a length-invariant token-presence
  algorithm when single-length shortcuts are less useful
- separate an architectural limitation from a training-distribution limitation

Main questions:

- does sparse exactly-one positive accuracy remain stable beyond length 900?
- does the positive logit margin stop drifting downward with length?
- do diagnostic slices show improvement specifically on exactly-one positives, rather
  than only on many-target positives?

Implementation constraint:

- do not introduce padding or masks in the first Stage 2A run
- keep each mini-batch single-length and alternate across length-specific loaders

## Stage 2B: Pooling Ablation

Keep the transformer fixed and compare:

- max pooling
- mean pooling
- sum pooling
- CLS-token pooling

Goal:

- isolate how sequence reduction changes extrapolation

Expectation:

- max pooling should be strongest
- mean pooling is the most likely to fail due to signal dilution

## Stage 3: Controlled Complexity

Only after the first stages are stable, vary:

- number of layers
- number of heads
- positional encoding type
- broader train-length distributions

Candidate values:

- layers: `1, 2, 4`
- heads: `1, 2, 4`
- positional encoding: `none, sinusoidal, learned`
- train lengths: `fixed 10`, `[10, 20, 50, 100]`, `uniform 5..100`

## Evaluation Plan

For every model, evaluate on:

- train length: `10`
- longer lengths: `20, 50, 100, 200, 500, 1000`

Track at least:

- overall accuracy by length
- accuracy on positives with exactly one target token
- accuracy on negatives

The primary evaluation distribution should use exactly-one positives and zero-target negatives at every length. Multiple-target positives should be reported as a secondary diagnostic slice after the primary setup is stable, not mixed into the initial training distribution.

For the transformer, also track:

- attention maps on selected examples
- attention entropy as length grows
- pooled activation magnitudes by length

## Main Hypotheses

1. The max-pooling baseline will generalize almost perfectly.
2. A tiny transformer with max pooling and no positional encoding will generalize reasonably well, though not necessarily perfectly.
3. Mean pooling will often fail at long lengths even if it reaches near-100% accuracy at length 10.
4. Learned absolute positional embeddings will make extrapolation brittle.
5. Training on a distribution of lengths will improve extrapolation relative to training on a single fixed length.
6. Training or evaluating with many-target positives can make the task artificially easier and may hide failures on the sparse exactly-one case.

## Main Risks

- the first task may be too easy, making all models look good
- the first task may be solved for the wrong reason, so interpretability must accompany accuracy
- evaluation at very long lengths may become slow if attention remains quadratic

## Deliverables

- `PLAN.md`
- `TASK.md`
- a small standalone PyTorch codebase for data generation, models, training, and evaluation
- plots of accuracy vs length
- qualitative attention visualizations for transformer runs
- an initial experiment report comparing the baseline and minimal transformer

## Immediate Next Step

Run Stage 2A.

That means:

1. keep the Stage 1 transformer architecture unchanged
2. train on `[10, 20, 50, 100]` using single-length batches
3. evaluate on the full length sweep
4. compare sparse exactly-one positive accuracy against Stage 1
5. inspect whether positive logits and attention summaries become more length-stable
