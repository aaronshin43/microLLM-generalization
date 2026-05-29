# Stage 1 Numerical Analysis Task

## Objective

Explain why the Stage 1 length-10-trained transformer classifies long exactly-one positive sequences as negative.

The goal is not to train a new model. The goal is to inspect the already trained Stage 1 checkpoint and identify where the target-token signal is lost as evaluation length increases.

## Target Model

Use the Stage 1 minimal transformer checkpoint:

- 1 transformer encoder layer
- 1 attention head
- `d_model = 64`
- no positional encoding
- max pooling
- trained only on length 10

Primary run directory:

```text
runs/stage1_transformer_maxpool
```

## Core Question

Why does the model solve length-10 token presence detection but fail on long exactly-one positives?

Working hypothesis:

> The model learns a finite target-score advantage that works at length 10, but this advantage does not scale with sequence length. As length grows, the attention softmax denominator increases, target attention mass decreases, target-aligned evidence weakens after the transformer block, and the final max-pooled classifier logit crosses below zero.

## Analysis Principles

Do not analyze attention weights alone. The full path to inspect is:

```text
tokens
-> embeddings
-> Q/K/V projections
-> attention logits
-> attention softmax
-> attention output
-> output projection
-> residual + layer norm
-> feedforward block
-> residual + layer norm
-> max pooling
-> classifier logit
```

The numerical analysis should connect attention behavior to the final logit collapse.

## Required Length Sweep

Analyze controlled exactly-one positive and zero-target negative examples at:

```text
10, 50, 100, 500, 700, 800, 850, 900, 950, 1000, 1100
```

## Analysis 1: Parameter Inspection

Extract and summarize trainable parameters from the trained checkpoint.

For each parameter, record:

- name
- shape
- parameter count
- L2 norm
- mean
- standard deviation
- min and max

For the attention layer, split PyTorch `in_proj_weight` and `in_proj_bias` into:

- `W_Q`, `b_Q`
- `W_K`, `b_K`
- `W_V`, `b_V`

Also inspect:

- `out_proj.weight`
- `out_proj.bias`
- feedforward weights
- layer norm parameters
- classifier weight and bias

Purpose:

Determine whether any obvious parameter scale or projection geometry explains the length-sensitive behavior.

## Analysis 2: Token-Level Q/K/V Geometry

Compute Q, K, and V vectors for the target token and every non-target vocabulary token.

For each token type, compute:

- `||q||`
- `||k||`
- `||v||`
- cosine similarity to the target token in Q space
- cosine similarity to the target token in K space
- cosine similarity to the target token in V space

Then compute pairwise attention scores:

```text
score(query_token, key_token) = q_query · k_key / sqrt(d_head)
```

Important comparisons:

- non-target query to target key
- target query to target key
- non-target query to non-target key
- target query to non-target key

Purpose:

Check whether the trained model actually separates the target token in Q/K/V space, and whether the target key receives a meaningful score advantage before softmax.

## Analysis 3: Attention Logit Decomposition By Length

For controlled exactly-one positive sequences, manually compute the attention logits:

```text
QK^T / sqrt(d_head)
```

For each length, compute:

- target key score mean across queries
- target key score max across queries
- non-target key score mean
- non-target key score max
- target score rank among all keys
- score gap: `target_score - max_non_target_score`
- score gap: `target_score - mean_non_target_score`

Run this for:

- all query positions
- target query position only
- last query position
- the query position with maximum attention to the target

Purpose:

Determine whether the target token is distinguishable before softmax, and whether the issue is score separation or softmax competition.

## Analysis 4: Softmax Denominator And Length Scaling

For each query row, decompose target attention mass:

```text
target_mass = exp(target_score) / sum_j exp(score_j)
```

Record:

- `exp(target_score)`
- `sum_non_target_exp_scores`
- full softmax denominator
- target mass
- maximum target mass over query rows
- mean target mass over query rows

Compare the observed target mass with the approximation:

```text
target_mass ≈ exp(delta) / (exp(delta) + L - 1)
```

where:

```text
delta = target_score - typical_non_target_score
```

Purpose:

Test whether the target score advantage is too small to survive the increase in competing non-target keys as length grows.

## Analysis 5: Attention Output And Residual Path

For each controlled sequence length, capture:

- embedding output
- attention weighted value output before `out_proj`
- attention output after `out_proj`
- hidden state after first residual + layer norm
- feedforward output
- hidden state after second residual + layer norm

For each captured tensor, compute:

- target position representation norm
- non-target representation norm mean and max
- cosine similarity between target and non-target representations
- classifier-aligned evidence:

```text
evidence_i = hidden_i · classifier_weight
```

Purpose:

Identify whether the target signal disappears during attention, output projection, residual mixing, layer norm, feedforward, or only after max pooling.

## Analysis 6: Max-Pool Source Attribution

The current classifier uses max pooling over sequence positions:

```text
pooled[d] = max_i hidden[i, d]
```

For each sequence, compute:

- argmax source position for every pooled dimension
- fraction of pooled dimensions sourced from the target position
- fraction sourced from non-target positions
- classifier-weighted contribution by source position

For classifier-weighted contribution, inspect:

```text
classifier_weight[d] * pooled[d]
```

and group dimensions by whether their max came from:

- target position
- non-target position

Purpose:

Test whether long sequences cause non-target extreme activations to dominate the max-pooled vector and reduce the final positive logit.

## Analysis 7: Logit Decomposition

Decompose the final classifier logit:

```text
logit = classifier_weight · pooled + classifier_bias
```

For each length, record:

- final logit
- classifier bias
- sum of positive-dimension contributions
- sum of negative-dimension contributions
- top positive contributing dimensions
- top negative contributing dimensions
- whether top dimensions are sourced from target or non-target positions

Purpose:

Connect representation changes directly to the decision boundary crossing at long lengths.

## Notebook Deliverable

Create a notebook:

```text
notebooks/stage1_attention_mechanism.ipynb
```

Recommended notebook sections:

1. Load checkpoint and config
2. Print trainable parameter descriptions
3. Split Q/K/V/O weights
4. Token-level Q/K/V geometry
5. Controlled sequence generation
6. Manual attention calculation
7. Softmax denominator analysis
8. Hidden-state and classifier-aligned evidence analysis
9. Max-pool source attribution
10. Logit decomposition
11. Summary of failure mechanism

Reusable analysis helpers should go in:

```text
src/analysis.py
```

Do not put all logic directly in the notebook if it can be reused or tested.

## Expected Outcome

The analysis should produce a short conclusion answering:

- Does the target token have a finite attention score advantage?
- Does target attention mass decay because the softmax denominator grows with length?
- Does the target signal remain visible after the transformer block?
- Does max pooling preserve or distort target evidence?
- Which numerical quantity best predicts the positive logit collapse?

The desired final output is a mechanistic explanation of why Stage 1 fails on long exactly-one positives despite solving the task at length 10.
