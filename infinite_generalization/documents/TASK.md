# Task: Empirical Validation Of The Simplified Length-Aware Attention Model

## Objective

Implement the simplified length-aware attention model described in `SIMPLIFIED_LENGTH_AWARE_ATTENTION_MODEL.md` and test whether the theoretical prediction matches empirical behavior when the model is actually trained.

The central question is:

```text
Does the theory match practice when this exact simplified model is used for the learning process?
```

This task should produce a small, controlled experiment that is much simpler than the full Stage 1 or Stage 2B transformer. The goal is to isolate the attention dilution mechanism and directly compare:

- theoretical target attention mass
- measured target attention mass from the trained model
- learned target-vs-non-target score margin
- binary classification accuracy by sequence length

## Motivation

The simplified theory assumes a two-token vocabulary:

- $t$: target token
- $u$: non-target token

For a positive sequence of length $n$:

```text
t, u, u, ..., u
```

the last-query attention score row is assumed to have the form:

```math
S_n = (a, b, b, \ldots, b),
\qquad
a > b.
```

The target-vs-non-target score margin is:

```math
\Delta = a-b.
```

After applying inverse temperature $\alpha$, the theoretical target attention mass is:

```math
p_t(n)
=
\frac{e^{\alpha\Delta}}
{e^{\alpha\Delta} + (n-1)}.
```

The theory predicts:

- if $\alpha$ is constant, $p_t(n) \to 0$
- if $\alpha = \log n$, the limit depends on $\Delta$
- if $\alpha = c\log n$, the limit depends on $c\Delta$

The empirical experiment should test whether this formula still describes the behavior when $a$, $b$, and $\Delta$ are produced by a trained model rather than manually assigned.

## Scope

This is a new controlled experiment, not a replacement for Stage 1 or Stage 2B.

The experiment should be implemented inside:

```text
infinite_generalization/
```

Recommended naming:

```text
src/stage3_simplified_attention.py
```

Recommended output directory:

```text
runs/stage3_simplified_attention/
```

The code should remain independent from the full transformer model unless a small shared utility is clearly useful.

## Core Experimental Setup

### Vocabulary

Use exactly two tokens:

- target token $t$
- non-target token $u$

Recommended token ids:

```text
t = 0
u = 1
```

### Input Distributions

Use two primary classes.

Positive example:

```text
t, u, u, ..., u
```

Negative example:

```text
u, u, u, ..., u
```

The first implementation should keep the target at position 0 for positive examples. This matches the theoretical setup exactly.

Optional later extension:

- target near beginning
- target near middle
- target near end
- random target position

Do not include these optional variants in the first implementation unless the exact-position experiment is already working.

### Embeddings

Start with fixed one-hot embeddings:

```math
t \mapsto [1,0],
\qquad
u \mapsto [0,1].
```

This keeps the experiment maximally close to the simplified theory.

Optional later extension:

- learned embeddings

The first implementation should prefer fixed embeddings so that the score mechanism is easier to interpret.

## Model Design

The model should implement a minimal last-query attention classifier.

### Input

For a sequence of length $n$, the embedded input is:

```math
X_n \in \mathbb{R}^{n \times 2}.
```

Variables used here:

- $n$ is sequence length.
- $X_n$ is the embedded input matrix.
- each row is either $[1,0]$ for $t$ or $[0,1]$ for $u$.

### Query, Key, And Value

Use a single attention query from the last token position.

Compute:

```math
Q = X_n W_Q,
\qquad
K = X_n W_K,
\qquad
V = X_n.
```

The last query is:

```math
q_{\mathrm{last}} = Q_n.
```

Variables used here:

- $W_Q$ is a learned query projection.
- $W_K$ is a learned key projection.
- $V=X_n$ means values are the original one-hot embeddings.
- $q_{\mathrm{last}}$ is the query vector from the last token.
- $Q_n$ is the query vector at the last position.

Rationale:

The theory reads only the last token output. Because the last token is $u$ in both positive and negative examples, the query is stable and the model must distinguish sequences through the keys.

### Attention Scores

For key position $j$, compute:

```math
s_j
=
\frac{q_{\mathrm{last}} \cdot k_j}{\sqrt{d_h}}.
```

For a positive sequence, measure:

```math
a = s_{\mathrm{target}},
\qquad
b = \frac{1}{n-1}\sum_{j \ne \mathrm{target}} s_j,
\qquad
\Delta = a-b.
```

Variables used here:

- $s_j$ is the raw attention score from the last query to key position $j$.
- $k_j$ is the key vector at position $j$.
- $d_h$ is the key/query dimension.
- $a$ is the target key score.
- $b$ is the mean non-target key score.
- $\Delta$ is the measured score margin.

Also measure non-target score variation:

```math
\operatorname{std}_{j \ne \mathrm{target}}(s_j).
```

This checks whether the empirical scores really match the theoretical form:

```math
(a,b,b,\ldots,b).
```

### Length-Aware Scaling

Support at least three inverse-temperature schedules:

```math
\alpha(n) = 1
```

```math
\alpha(n) = \log n
```

```math
\alpha(n) = c\log n
```

where $c$ can be fixed or learned.

Recommended first implementation:

- `constant`: $\alpha(n)=1$
- `log`: $\alpha(n)=\log n$
- `learned_log`: $\alpha(n)=1+\mathrm{softplus}(k_\alpha)\log(1+n)$

Variables used here:

- $\alpha(n)$ is the length-dependent score multiplier.
- $c$ is a scalar coefficient.
- $k_\alpha$ is a learned scalar parameter.
- $\mathrm{softplus}(x)=\log(1+e^x)$, which ensures a positive learned coefficient.

### Attention Weights

The corrected scores are:

```math
\tilde{s}_j = \alpha(n)s_j.
```

The attention weights are:

```math
A_j
=
\frac{e^{\tilde{s}_j}}
{\sum_{\ell=1}^{n} e^{\tilde{s}_\ell}}.
```

The empirical target attention mass is:

```math
p_{\mathrm{emp}}(n) = A_{\mathrm{target}}.
```

Variables used here:

- $\tilde{s}_j$ is the scaled attention score.
- $A_j$ is the attention weight assigned to key position $j$.
- $p_{\mathrm{emp}}(n)$ is the measured target attention mass.

### Attention Output

Use one-hot values:

```math
V = X_n.
```

The attention output is:

```math
o(n)
=
\sum_{j=1}^{n} A_j V_j.
```

For a positive sequence with the target at position 0, this should be:

```math
o(n)
=
(p_{\mathrm{emp}}(n), 1-p_{\mathrm{emp}}(n)).
```

Variables used here:

- $o(n)$ is the last-query attention output.
- $V_j$ is the value vector at position $j$.
- $A_j$ is the attention weight for position $j$.
- the first coordinate is target mass because $t \mapsto [1,0]$.
- the second coordinate is non-target mass because $u \mapsto [0,1]$.

### Classifier

Use a small linear classifier:

```math
z(n)
=
w^\top o(n) + b_{\mathrm{cls}}.
```

Variables used here:

- $z(n)$ is the final binary-classification logit.
- $w$ is a learned classifier weight vector.
- $b_{\mathrm{cls}}$ is a learned classifier bias.
- $o(n)$ is the attention output.

Prediction rule:

```math
\hat{y}=1
\quad
\text{if}
\quad
z(n) \ge 0.
```

## Training Setup

### Training Lengths

Start with fixed short-length training:

```text
train_length = 10
```

Optional second condition:

```text
train_lengths = [10, 20, 50, 100]
```

The first goal is to understand the fixed-length case.

### Evaluation Lengths

Evaluate on a broad length sweep:

```text
10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000
```

Optional extension:

```text
20000, 50000, 100000
```

Only add very long lengths after the basic experiment is working.

### Loss

Use binary cross entropy with logits:

```math
L(y,z)
=
-
y\log\sigma(z)
-
(1-y)\log(1-\sigma(z)).
```

Variables used here:

- $L$ is the loss.
- $y$ is the true label.
- $z$ is the final logit.
- $\sigma(z)$ is the sigmoid probability.

## Theoretical Prediction To Compare Against

For each positive evaluation example, compute the empirical margin:

```math
\Delta_{\mathrm{emp}}(n)
=
a(n)-b(n).
```

Then compute the theory-predicted target attention mass:

```math
p_{\mathrm{theory}}(n)
=
\frac{e^{\alpha(n)\Delta_{\mathrm{emp}}(n)}}
{e^{\alpha(n)\Delta_{\mathrm{emp}}(n)} + (n-1)}.
```

Variables used here:

- $\Delta_{\mathrm{emp}}(n)$ is the measured target-vs-mean-non-target score margin.
- $p_{\mathrm{theory}}(n)$ is the theoretical prediction using the measured margin.
- $\alpha(n)$ is the same length-dependent scale used by the model.

Also compute a fixed-margin theory curve using the training-length margin:

```math
p_{\mathrm{theory,train}}(n)
=
\frac{e^{\alpha(n)\Delta_{\mathrm{train}}}}
{e^{\alpha(n)\Delta_{\mathrm{train}}} + (n-1)}.
```

Variables used here:

- $\Delta_{\mathrm{train}}$ is the measured margin at training length.
- This curve tests whether the model learned an approximately fixed margin.

## Metrics To Save

Save a CSV such as:

```text
runs/stage3_simplified_attention/metrics_by_length.csv
```

Required columns:

- `length`
- `split`
- `alpha_mode`
- `accuracy`
- `positive_accuracy`
- `negative_accuracy`
- `mean_logit_positive`
- `mean_logit_negative`
- `mean_probability_positive`
- `mean_probability_negative`
- `mean_target_score_a`
- `mean_non_target_score_b`
- `mean_delta`
- `std_non_target_scores`
- `mean_empirical_target_attention`
- `mean_theory_target_attention_using_empirical_delta`
- `mean_theory_target_attention_using_train_delta`
- `mean_attention_absolute_error_empirical_vs_theory`

Recommended additional columns:

- `learned_alpha_coefficient`
- `classifier_weight_target_coord`
- `classifier_weight_non_target_coord`
- `classifier_bias`

Coordinate convention:

- target coordinate means the first coordinate because $t=0$ and $t \mapsto [1,0]$.
- non-target coordinate means the second coordinate because $u=1$ and $u \mapsto [0,1]$.

## Figures To Produce

Create figures under:

```text
infinite_generalization/documents/figures/
```

or run-specific figures under:

```text
runs/stage3_simplified_attention/figures/
```

Required figures:

### 1. Theory vs Empirical Target Attention

Plot by length:

- empirical target attention mass
- theory prediction using empirical $\Delta(n)$
- theory prediction using training-length $\Delta_{\mathrm{train}}$

Expected file:

```text
stage3_theory_vs_empirical_attention.png
```

### 2. Measured Margin By Length

Plot:

```math
\Delta_{\mathrm{emp}}(n)
```

by length.

Expected file:

```text
stage3_delta_by_length.png
```

### 3. Non-Target Score Variation

Plot:

```math
\operatorname{std}_{j \ne \mathrm{target}}(s_j)
```

by length.

Expected file:

```text
stage3_non_target_score_std.png
```

This checks whether the exact theory assumption $(a,b,b,\ldots,b)$ is actually satisfied.

### 4. Accuracy And Logit By Length

Plot:

- accuracy by length
- positive logit by length
- negative logit by length

Expected file:

```text
stage3_accuracy_and_logits.png
```

## Expected Outcomes

### Case 1: Theory Matches Practice

The theory is strongly supported if:

- non-target score standard deviation is near zero
- measured $\Delta(n)$ is approximately stable across length
- empirical target attention matches $p_{\mathrm{theory}}(n)$
- classification behavior changes exactly when target attention becomes too small

Interpretation:

The simplified model accurately explains the learned model because the learned model preserves the two-score structure assumed by the theory.

### Case 2: Theory Partially Matches Practice

The theory is partially supported if:

- empirical attention follows the theoretical trend
- but $\Delta(n)$ changes with length
- or non-target scores are not exactly equal
- or classifier behavior depends on more than target attention mass

Interpretation:

The softmax denominator mechanism is real, but the learned model introduces additional effects.

### Case 3: Theory Does Not Match Practice

The theory fails for this exact trained model if:

- empirical target attention is far from the theoretical prediction
- non-target scores vary strongly
- the learned query/key system does not produce the assumed $(a,b,b,\ldots,b)$ structure
- classification succeeds or fails for reasons unrelated to target attention mass

Interpretation:

The theoretical formula may still be mathematically correct, but the trained model does not realize the assumptions needed for the formula to describe its behavior.

## Implementation Steps

### Step 1: Add The Stage 3 Script

Create:

```text
src/stage3_simplified_attention.py
```

The script should:

- parse command-line arguments
- create train/eval datasets
- define the simplified attention model
- train the model
- evaluate by length
- save CSV metrics
- save figures when `matplotlib` is available

### Step 2: Implement The Dataset

Add a small local dataset generator inside the Stage 3 script unless reuse is clearly cleaner.

Requirements:

- generate positives as `[t, u, ..., u]`
- generate negatives as `[u, u, ..., u]`
- balance positives and negatives
- support deterministic seeds
- support arbitrary sequence lengths

### Step 3: Implement The Model

The first model should use:

- fixed one-hot token embeddings
- learned $W_Q$
- learned $W_K$
- values equal to one-hot embeddings
- last-query attention
- one linear classifier

The model forward pass should optionally return analysis details:

- raw scores
- corrected scores
- attention weights
- target score
- non-target scores
- margin
- attention output
- final logit

### Step 4: Train And Evaluate

Train on length 10 first.

Evaluate on:

```text
10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000
```

Run at least:

- constant $\alpha$
- $\alpha=\log n$
- learned log scale

### Step 5: Compare Theory And Practice

For each length:

1. Measure empirical target attention mass.
2. Measure $a$, $b$, and $\Delta$.
3. Compute theory prediction using empirical $\Delta(n)$.
4. Compute theory prediction using training-length $\Delta_{\mathrm{train}}$.
5. Save the absolute difference between empirical and theory attention.

### Step 6: Document Results

After running the experiment, add a new report:

```text
documents/STAGE3_SIMPLIFIED_ATTENTION_EMPIRICAL_ANALYSIS.md
```

The report should answer:

- Did the learned model produce the assumed score structure $(a,b,b,\ldots,b)$?
- Did empirical target attention match the theoretical formula?
- Did $\alpha=\log n$ behave according to the learned $\Delta$?
- Did learned log scaling increase the effective margin enough?
- What does this imply for Stage 2B?

## Smoke Test Requirements

Add or run a small smoke test that verifies:

- the script runs with very few steps
- output CSV is created
- evaluation includes at least two lengths
- theory and empirical attention columns are present
- no NaN values appear in saved metrics

Recommended command:

```text
$env:PYTHONPATH = 'src'; ..\.venv\Scripts\python.exe src\stage3_simplified_attention.py --smoke-test
```

Use the exact command style already used in `README.md` if different.

## Success Criteria

This task is complete when:

- the simplified trainable attention model is implemented
- the model can train and evaluate on controlled two-token data
- length-sweep metrics are saved to CSV
- theory-vs-empirical attention figures are generated
- the result clearly states whether the theory matches practice
- the connection to Stage 2B is documented without overclaiming

## Notes And Risks

- If embeddings are learned, the model may no longer exactly match the professor's setup. Use fixed one-hot embeddings first.
- If the classifier learns to ignore the target-attention coordinate, accuracy may not reflect attention behavior. Save classifier weights.
- If non-target scores are not equal, the simple formula using mean $b$ is only an approximation. Save non-target score standard deviation.
- If $\Delta$ changes with length, the fixed-margin theory curve may not match, but the empirical-margin theory curve may still match.
- This experiment validates the simplified mechanism, not the full transformer.
