# Task: Mechanistic Interpretation Of The Simplest Model

## Objective

Analyze the trained Stage 3 simplest model at the weight level.

This is not a new training experiment. The goal is to inspect an already trained reduced attention model and explicitly show how its learned query and key projections create:

```math
a>b.
```

Here:

- $a$ is the attention score from the final query to the target key.
- $b$ is the attention score from the final query to a non-target key.
- $\Delta=a-b$ is the target-vs-non-target score margin.

The central question is:

**How do the learned $W_Q$ and $W_K$ matrices make the final query assign a larger score to the target token than to the non-target token?**

## Background

The simplest Stage 3 setup uses two tokens:

- target token $t=0$
- non-target token $u=1$

The values are fixed one-hot vectors:

```math
t \mapsto [1,0],
\qquad
u \mapsto [0,1].
```

Positive inputs have the form:

```text
t, u, u, ..., u
```

Negative inputs have the form:

```text
u, u, u, ..., u
```

The model uses:

- learned query projection $W_Q$
- learned key projection $W_K$
- fixed values equal to the one-hot inputs
- last-query attention
- final linear classifier on the attention output

Because positive examples always end in $u$, the final query is the query produced by the non-target token:

```math
q_{\mathrm{last}}=q_u.
```

## Core Calculation

Given the one-hot token vectors:

```math
x_t=[1,0],
\qquad
x_u=[0,1],
```

compute:

```math
q_u = W_Q x_u,
```

```math
k_t = W_K x_t,
\qquad
k_u = W_K x_u.
```

Then compute:

```math
a =
\frac{q_u^\top k_t}{\sqrt{d}},
```

```math
b =
\frac{q_u^\top k_u}{\sqrt{d}},
```

where $d$ is the attention head dimension.

Finally compute:

```math
\Delta=a-b.
```

The target score is larger when:

```math
\Delta>0.
```

## Mechanistic Decomposition

The most important identity is:

```math
\Delta
=
\frac{q_u^\top k_t-q_u^\top k_u}{\sqrt{d}}
=
\frac{q_u^\top(k_t-k_u)}{\sqrt{d}}.
```

Therefore, the model creates $a>b$ exactly when:

```math
q_u^\top(k_t-k_u)>0.
```

Interpretation:

- $q_u$ describes what the final non-target query is looking for.
- $k_t-k_u$ describes how the target key differs from the non-target key.
- If $q_u$ aligns positively with $k_t-k_u$, then the target key receives a higher score.

This is the main mechanism to explain.

## Required Outputs

Create a lightweight analysis script or notebook that loads a trained Stage 3 model and prints or saves:

- $W_Q$
- $W_K$
- $q_u$
- $k_t$
- $k_u$
- $k_t-k_u$
- component-wise $q_u \odot k_t$
- component-wise $q_u \odot k_u$
- component-wise $q_u \odot (k_t-k_u)$
- $a$
- $b$
- $\Delta$

The analysis should make clear which dimensions contribute positively or negatively to $\Delta$.

## Recommended Script

Add a script:

```text
infinite_generalization/src/analyze_stage3_mechanism.py
```

The script should accept:

```text
--run-dir runs/stage3_simplified_attention_learned_log_e200
```

or another Stage 3 run directory containing a trained model checkpoint.

If the current Stage 3 pipeline does not save model checkpoints, add checkpoint saving to:

```text
infinite_generalization/src/stage3_simplified_attention.py
```

Recommended checkpoint path:

```text
model.pt
```

inside each run directory.

## Analysis Steps

### Step 1: Confirm Checkpoint Availability

Check whether the Stage 3 runs save trained model weights.

If not, modify the Stage 3 training script to save:

```text
model.pt
```

with enough metadata to reload:

- `state_dict`
- `d_head`
- `alpha_mode`
- `alpha_log_scale_init`

### Step 2: Load A Trained Model

Start with the strongest simple learned-log run:

```text
runs/stage3_simplified_attention_learned_log_e200
```

If that checkpoint is not available, rerun the Stage 3 learned-log condition with:

```text
--alpha-mode learned_log
--train-lengths 10
--max-train-steps 6400
```

### Step 3: Extract Query And Key Weights

Load:

```math
W_Q,
\qquad
W_K.
```

Be careful with PyTorch shape conventions.

For:

```python
nn.Linear(2, d_head, bias=False)
```

the stored weight has shape:

```text
[d_head, 2]
```

Therefore:

```python
q_u = W_Q @ x_u
k_t = W_K @ x_t
k_u = W_K @ x_u
```

where:

```python
x_t = [1, 0]
x_u = [0, 1]
```

### Step 4: Compute Scores

Compute:

```math
a =
\frac{q_u^\top k_t}{\sqrt{d}},
\qquad
b =
\frac{q_u^\top k_u}{\sqrt{d}},
\qquad
\Delta=a-b.
```

Verify that the computed $\Delta$ matches the model's `mean_delta` in `metrics_by_length.csv`.

### Step 5: Decompose The Margin

Compute:

```math
q_u \odot (k_t-k_u).
```

Then:

```math
\Delta
=
\frac{\sum_i q_{u,i}(k_{t,i}-k_{u,i})}{\sqrt{d}}.
```

Report each dimension's contribution.

This explains which hidden dimensions create the positive target-vs-non-target score margin.

### Step 6: Write A Short Mechanistic Note

Add a short section to:

```text
infinite_generalization/documents/STAGE3_SIMPLIFIED_LENGTH_AWARE_ATTENTION.md
```

The note should explain:

- the final query is $q_u$ because the last token is $u$
- the model creates $a>b$ because $q_u$ aligns more with $k_t$ than $k_u$
- equivalently, $q_u^\top(k_t-k_u)>0$
- the component-wise decomposition shows which dimensions create the margin

## Success Criteria

This task is complete when:

- a trained Stage 3 model checkpoint can be loaded
- $W_Q$ and $W_K$ are extracted
- $q_u$, $k_t$, and $k_u$ are computed directly from the weights
- $a$, $b$, and $\Delta$ are computed directly
- the computed $\Delta$ matches the recorded Stage 3 metric
- the margin is decomposed by dimension
- the report explains why $a>b$ in weight-level terms

## Notes

- This analysis applies to the simplest reduced model only.
- Do not add full transformer components in this task.
- Do not change the target position.
- Do not change token values or embeddings.
- The purpose is interpretability, not improving performance.
