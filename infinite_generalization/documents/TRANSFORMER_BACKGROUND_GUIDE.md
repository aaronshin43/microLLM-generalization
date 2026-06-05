# Transformer Background Guide For This Project

## Purpose

This guide explains the machine learning and transformer concepts needed to understand the length-generalization experiments in this repository.

The goal is not to cover all of machine learning. The goal is to build enough background to read the Stage 1, Stage 2B, and theoretical analysis documents with confidence.

The central research question is:

```text
Can a small model trained on short sequences learn a rule that works for much longer sequences?
```

In this project, the task is intentionally simple:

```text
Given a sequence of tokens, predict whether the target token is present.
```

This is a binary classification problem.

## Definitions And Notation

This section defines basic terms used throughout the guide. Some definitions are repeated later near equations so each section can be read independently.

- Model: a function with trainable parameters that maps an input to an output.
- Classifier: a model or model component that assigns an input to a category. In this project, the classifier predicts target present or target absent.
- Binary classification: classification with two possible labels, usually written as 0 and 1.
- Input: the data given to the model. Here, the input is a sequence of token ids.
- Label: the correct answer for an input. Here, label 1 means target present and label 0 means target absent.
- Prediction: the answer produced by the model.
- Token: a discrete symbol in a sequence.
- Vocabulary: the set of all possible tokens.
- Token id: an integer representing a token.
- Embedding: a learned vector representation of a token.
- Parameter: a trainable number inside the model, such as a weight or bias.
- Weight: a parameter that multiplies an input value or feature.
- Bias: a parameter added after weighted features are combined.
- Logit: the raw scalar output used for binary classification before applying sigmoid.
- Sigmoid: a function that maps a logit to a value between 0 and 1.
- Probability: in this guide, the sigmoid-transformed logit.
- Loss: a scalar number measuring how wrong the model is on an example or batch.
- Binary cross entropy: the standard loss for binary classification.
- Gradient: the derivative of the loss with respect to model parameters.
- Backpropagation: the algorithm that computes gradients through the model.
- Optimizer: the algorithm that updates parameters using gradients.
- Learning rate: the step size used by the optimizer.
- Batch: a small group of training examples processed together.
- Epoch: one pass through the training dataset.
- Train set: examples used to update model parameters.
- Validation set: examples used to monitor training without directly updating parameters.
- Test set: examples used for final evaluation.
- Accuracy: the fraction of examples classified correctly.
- Generalization: performance on examples not used for training.
- Extrapolation: generalization outside the training range, such as testing on much longer sequences.
- Overfitting: doing well on training data but poorly on new data.
- Shortcut learning: learning a rule that works on training data but is not the intended algorithm.
- Attention score: an unnormalized value measuring how much one token position wants to read from another.
- Attention weight: a normalized attention score after softmax.
- Softmax: a function that converts a list of scores into nonnegative weights that sum to 1.
- Softmax denominator: the sum of exponentiated scores in the bottom of the softmax fraction.
- Entropy: a measure of how spread out a probability distribution is. Higher attention entropy means attention is more diffuse.
- Activation: a value inside the model after some computation.
- Pooling: a method for converting many token-level vectors into one sequence-level vector.
- Margin: the difference between a target score and a competing non-target score.
- Intervention: a deliberate architecture or scoring modification used to test a hypothesis.
- Diagnostic slice: a controlled subset of evaluation examples, such as exactly-one-target positives or target-near-end positives.

Common notation:

- $n$: sequence length.
- $x$: input sequence.
- $x_i$: token at position $i$.
- $y$: true label.
- $\hat{y}$: predicted label.
- $z$: final classifier logit.
- $p$: probability after sigmoid.
- $\theta$: all trainable model parameters.
- $L$: loss.
- $\eta$: learning rate.
- $V$: vocabulary size.
- $d$: embedding or model dimension.
- $X$: embedded input matrix.
- $Q$, $K$, $V$: query, key, and value matrices. The symbol $V$ can mean vocabulary size or value matrix depending on context.
- $q_i$, $k_j$, $v_j$: query, key, and value vectors for token positions.
- $s_{ij}$: raw attention score from query position $i$ to key position $j$.
- $A_{ij}$: attention weight from query position $i$ to key position $j$.
- $\Delta$: score margin, usually target score minus non-target score.
- $\alpha$: inverse temperature or score scaling factor.
- $\beta$: additive score-bias scale.
- $r_j$: learned target-likeness score for key position $j$.

## 1. Binary Classification

In binary classification, each input has a label:

```math
y \in \{0, 1\}.
```

For this project:

```math
y = 1
```

means the target token is present, and:

```math
y = 0
```

means the target token is absent.

The model receives an input sequence:

```math
x = (x_1, x_2, \ldots, x_n),
```

where $n$ is the sequence length. The model outputs one real number called a logit:

```math
z \in \mathbb{R}.
```

Variables used here:

- $x$ is the input sequence.
- $x_i$ is the token at position $i$.
- $n$ is the number of tokens in the sequence.
- $y$ is the true binary label.
- $z$ is the model's raw scalar output, called the logit.

The decision rule is:

```math
\hat{y}
=
\begin{cases}
1, & z \ge 0 \\
0, & z < 0.
\end{cases}
```

Variables used here:

- $\hat{y}$ is the predicted label.
- $z$ is the logit.
- The threshold $z=0$ corresponds to probability 0.5 after sigmoid.

Interpretation:

The logit is the model's raw evidence. Positive logits mean the model predicts target present. Negative logits mean the model predicts target absent.

## 2. Logit, Sigmoid, And Probability

The sigmoid function converts a logit into a number between 0 and 1:

```math
\sigma(z)
=
\frac{1}{1+e^{-z}}.
```

Variables used here:

- $\sigma$ is the sigmoid function.
- $z$ is the logit.
- $e$ is Euler's number, approximately 2.718.

The model probability is:

```math
p = \sigma(z).
```

Variables used here:

- $p$ is the model's predicted probability for label 1.
- $\sigma(z)$ is the sigmoid-transformed logit.

If $z = 0$, then:

```math
p = 0.5.
```

If $z > 0$, then:

```math
p > 0.5.
```

If $z < 0$, then:

```math
p < 0.5.
```

This is why the classification threshold is $z = 0$.

Interpretation:

The probability is a transformed version of the logit. The logit is usually more useful for mechanism analysis because it is the quantity directly produced by the final classifier.

## 3. Loss Function

During training, the model is optimized to make correct predictions. For binary classification, the usual loss is binary cross entropy:

```math
L(y,z)
=
-
y\log\sigma(z)
-
(1-y)\log(1-\sigma(z)).
```

Variables used here:

- $L(y,z)$ is the loss for one example.
- $y$ is the true label, either 0 or 1.
- $z$ is the logit.
- $\sigma(z)$ is the predicted probability of label 1.
- $\log$ is the natural logarithm.

For a positive example where $y=1$:

```math
L(1,z)
=
-
\log\sigma(z).
```

The loss becomes small when $z$ is large and positive.

For a negative example where $y=0$:

```math
L(0,z)
=
-
\log(1-\sigma(z)).
```

The loss becomes small when $z$ is large and negative.

Interpretation:

Training pushes positive examples toward positive logits and negative examples toward negative logits.

## 4. Parameters, Gradients, And Optimization

A model contains trainable parameters such as weights and biases. Denote all parameters by:

```math
\theta.
```

Variables used here:

- $\theta$ denotes all trainable model parameters together.

Training minimizes average loss:

```math
\min_\theta
\frac{1}{m}
\sum_{i=1}^{m}
L(y_i, f_\theta(x_i)).
```

Variables used here:

- $m$ is the number of training examples in the average.
- $i$ indexes training examples.
- $x_i$ is the input for example $i$.
- $y_i$ is the true label for example $i$.
- $f_\theta$ is the model with parameters $\theta$.
- $L(y_i,f_\theta(x_i))$ is the loss on example $i$.

Gradient descent updates parameters in the direction that reduces loss:

```math
\theta
\leftarrow
\theta
-
\eta
\nabla_\theta L.
```

Variables used here:

- $\theta$ is the current parameter vector.
- $\eta$ is the learning rate.
- $\nabla_\theta L$ is the gradient of the loss with respect to $\theta$.
- The arrow means the parameter value is updated.

Interpretation:

The model is not explicitly programmed with the target-token rule. It learns parameter values that reduce training loss. This makes shortcut learning possible.

## 5. Generalization And Extrapolation

Generalization means the model works on examples it did not train on.

Length extrapolation is a harder kind of generalization:

```text
Train on short sequences, then test on much longer sequences.
```

In this project, a model may train on length 10 and be evaluated on lengths such as 100, 1000, or 10000.

The desired rule is length-invariant:

```text
Return positive if the target token appears anywhere, regardless of sequence length.
```

Interpretation:

High accuracy at training length does not prove that the model learned the true algorithm. It may have learned a shortcut that only works near the training length.

## 6. Tokens And Vocabulary

A vocabulary is the set of possible tokens. Each token is assigned an integer id.

For example:

```text
target token: T
non-target token: A, B, C, ...
```

The input sequence is a list of token ids:

```math
(x_1, x_2, \ldots, x_n).
```

Variables used here:

- $x_i$ is the token id at position $i$.
- $n$ is the sequence length.

The model must convert token ids into vectors before doing neural computation.

## 7. Embeddings

An embedding layer maps each token id to a vector:

```math
E:
\{1,\ldots,V\}
\to
\mathbb{R}^{d}.
```

Variables used here:

- $E$ is the embedding lookup function or embedding table.
- $V$ is vocabulary size.
- $d$ is embedding dimension.
- $\mathbb{R}^{d}$ is the space of real-valued vectors with $d$ components.

The embedded sequence is:

```math
X
=
\begin{bmatrix}
e(x_1) \\
e(x_2) \\
\vdots \\
e(x_n)
\end{bmatrix}
\in
\mathbb{R}^{n \times d}.
```

Variables used here:

- $X$ is the embedded sequence matrix.
- $e(x_i)$ is the embedding vector for token $x_i$.
- $n$ is sequence length.
- $d$ is embedding dimension.
- $\mathbb{R}^{n \times d}$ means a real-valued matrix with $n$ rows and $d$ columns.

Interpretation:

The model does not process token names directly. It processes learned vectors. If the model learns that the target token has a special embedding direction, later layers can use that direction as evidence.

## 8. Attention: Query, Key, Value

Self-attention lets each token position look at other token positions.

Given an embedded sequence:

```math
X \in \mathbb{R}^{n \times d},
```

Variables used here:

- $X$ is the embedded sequence matrix.
- $n$ is sequence length.
- $d$ is model dimension.

attention forms three matrices:

```math
Q = XW_Q,
\qquad
K = XW_K,
\qquad
V = XW_V.
```

Variables used here:

- $Q$ contains query vectors.
- $K$ contains key vectors.
- $V$ contains value vectors.
- $W_Q$, $W_K$, and $W_V$ are learned projection matrices.
- $X$ is the embedded input matrix.

For query position $i$ and key position $j$, the raw attention score is:

```math
s_{ij}
=
\frac{q_i \cdot k_j}{\sqrt{d_h}}.
```

Variables used here:

- $s_{ij}$ is the raw attention score from query position $i$ to key position $j$.
- $q_i$ is the query vector at position $i$.
- $k_j$ is the key vector at position $j$.
- $q_i \cdot k_j$ is the dot product between those vectors.
- $d_h$ is the attention head dimension.

Interpretation:

$s_{ij}$ measures how strongly position $i$ wants to read from position $j$ before normalization.

## 9. Attention Weights And Softmax

Raw attention scores are converted into attention weights with softmax:

```math
A_{ij}
=
\frac{e^{s_{ij}}}
{\sum_{k=1}^{n} e^{s_{ik}}}.
```

Variables used here:

- $A_{ij}$ is the attention weight from query position $i$ to key position $j$.
- $s_{ij}$ is the raw attention score for key position $j$.
- $s_{ik}$ is the raw attention score for key position $k$.
- $k$ is a summation index over all key positions.
- $n$ is sequence length.
- The denominator normalizes the weights so they sum to 1.

For each query position $i$:

```math
\sum_{j=1}^{n} A_{ij} = 1.
```

Variables used here:

- $j$ indexes key positions.
- $A_{ij}$ is the attention weight assigned by query position $i$ to key position $j$.
- The sum equals 1 because softmax produces a probability distribution over keys.

The attention output at position $i$ is a weighted average of value vectors:

```math
o_i
=
\sum_{j=1}^{n}
A_{ij}v_j.
```

Variables used here:

- $o_i$ is the attention output vector at query position $i$.
- $A_{ij}$ is the attention weight on key/value position $j$.
- $v_j$ is the value vector at position $j$.
- $n$ is sequence length.

Interpretation:

Attention does not copy every token equally. It averages value vectors using weights determined by score similarity.

## 10. Softmax Denominator And Attention Dilution

The softmax denominator is:

```math
\sum_{k=1}^{n} e^{s_{ik}}.
```

Variables used here:

- $k$ indexes all key positions in the sequence.
- $n$ is sequence length.
- $s_{ik}$ is the raw attention score from query position $i$ to key position $k$.

This denominator grows when there are more tokens, even if each individual non-target token has a lower score than the target token.

Suppose one target key has score $a$, and each of the $n-1$ non-target keys has score $b$, with:

```math
a > b.
```

Variables used here:

- $a$ is the target key's raw attention score.
- $b$ is each non-target key's raw attention score in the simplified example.

The target attention mass is:

```math
p_t(n)
=
\frac{e^a}
{e^a + (n-1)e^b}.
```

Variables used here:

- $p_t(n)$ is the attention mass assigned to the target token at sequence length $n$.
- $a$ is the target score.
- $b$ is the non-target score.
- $n-1$ is the number of non-target tokens.

Let:

```math
\Delta = a-b.
```

Variables used here:

- $\Delta$ is the target-vs-non-target score margin.
- $a$ is the target score.
- $b$ is the non-target score.

Then:

```math
p_t(n)
=
\frac{e^\Delta}
{e^\Delta + (n-1)}.
```

Variables used here:

- $p_t(n)$ is target attention mass.
- $\Delta$ is the fixed score margin.
- $n$ is sequence length.

If $\Delta$ is fixed, then:

```math
p_t(n) \to 0
\qquad
\text{as } n \to \infty.
```

Interpretation:

The target can beat every non-target individually and still lose against all non-targets collectively. This is the core attention dilution problem.

## 11. Multi-Head Attention

A transformer usually uses multiple attention heads. Each head has its own $W_Q$, $W_K$, and $W_V$.

For head $h$:

```math
Q^{(h)} = XW_Q^{(h)},
\qquad
K^{(h)} = XW_K^{(h)},
\qquad
V^{(h)} = XW_V^{(h)}.
```

Variables used here:

- $h$ indexes the attention head.
- $Q^{(h)}$, $K^{(h)}$, and $V^{(h)}$ are query, key, and value matrices for head $h$.
- $W_Q^{(h)}$, $W_K^{(h)}$, and $W_V^{(h)}$ are learned projection matrices for head $h$.
- $X$ is the input representation.

Each head produces an output. These outputs are concatenated and passed through an output projection:

```math
\operatorname{MHA}(X)
=
\operatorname{Concat}(O^{(1)},\ldots,O^{(H)})W_O.
```

Variables used here:

- $\operatorname{MHA}(X)$ is the multi-head attention output.
- $O^{(h)}$ is the attention output from head $h$.
- $H$ is the number of attention heads.
- $W_O$ is the learned output projection matrix.

Interpretation:

Different heads can learn different reading patterns. In our experiments, head-level attention summaries help check whether any head specializes into a target detector.

## 12. Positional Encoding

Transformers often add positional information:

```math
X_{\mathrm{pos}}
=
X + P.
```

Variables used here:

- $X_{\mathrm{pos}}$ is the input representation after adding position information.
- $X$ is the token embedding matrix.
- $P$ is the positional encoding matrix.

In these experiments, the minimal transformer uses no positional encoding. This is intentional. The target-presence task does not require position information:

```text
The answer should not depend on where the target appears.
```

Interpretation:

Without positional encoding, the model is encouraged to learn a permutation-invariant rule. However, this does not guarantee length-invariant generalization.

## 13. Transformer Encoder Layer

A transformer encoder layer usually contains:

```text
self-attention
residual connection
layer normalization
feed-forward network
residual connection
layer normalization
```

A simplified view is:

```math
H
=
\operatorname{AttentionBlock}(X),
```

Variables used here:

- $H$ is the hidden representation after the attention block.
- $X$ is the input representation to the encoder layer.
- $\operatorname{AttentionBlock}$ represents self-attention plus surrounding layer components.

then:

```math
Z
=
\operatorname{FeedForwardBlock}(H).
```

Variables used here:

- $Z$ is the output of the encoder layer.
- $H$ is the input to the feed-forward block.
- $\operatorname{FeedForwardBlock}$ applies position-wise learned transformations.

Interpretation:

Attention controls how tokens exchange information. The feed-forward network transforms each position after information has been mixed.

## 14. Residual Connections

A residual connection adds the input back to the block output:

```math
Y
=
X + F(X).
```

Variables used here:

- $Y$ is the output after the residual connection.
- $X$ is the block input.
- $F(X)$ is the learned transformation applied by the block.

Interpretation:

Residual connections make optimization easier and let the model preserve input information while adding learned transformations.

## 15. Layer Normalization

LayerNorm normalizes activations within each token representation:

```math
\operatorname{LayerNorm}(x)
=
\gamma
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
+
\beta.
```

Variables used here:

- $x$ is one activation vector.
- $\mu$ is the mean of that vector's components.
- $\sigma^2$ is the variance of that vector's components.
- $\epsilon$ is a small constant for numerical stability.
- $\gamma$ and $\beta$ are learned scale and shift parameters.

Interpretation:

LayerNorm stabilizes training. It can also affect scale-based mechanisms, so it matters when interpreting logits and activation magnitudes.

## 16. Pooling

After the transformer encoder, the model has one vector per token position:

```math
H
\in
\mathbb{R}^{n \times d}.
```

Variables used here:

- $H$ is the token-level hidden representation after the transformer.
- $n$ is sequence length.
- $d$ is hidden dimension.

To classify the whole sequence, it needs one sequence-level vector.

Max pooling computes:

```math
h_{\mathrm{pool},k}
=
\max_{1 \le i \le n}
H_{ik}.
```

Variables used here:

- $h_{\mathrm{pool},k}$ is component $k$ of the pooled sequence vector.
- $H_{ik}$ is component $k$ of the hidden vector at token position $i$.
- $i$ indexes token positions.
- $n$ is sequence length.

This takes the maximum activation across positions for each dimension.

Interpretation:

Max pooling is useful for existential tasks because one strong target-token activation can be enough. But as length grows, non-target tokens get more chances to produce large accidental activations.

## 17. Final Classifier

The final classifier is usually a linear layer:

```math
z
=
w^\top h_{\mathrm{pool}} + b.
```

Variables used here:

- $h_{\mathrm{pool}}$ is the pooled sequence representation.
- $w$ is the learned classifier weight vector.
- $b$ is the learned classifier bias.
- $z$ is the final logit.
- $w^\top h_{\mathrm{pool}}$ is a dot product.

Interpretation:

The classifier converts pooled features into one scalar decision. If non-target pooled features grow with length, they can push the logit in the wrong direction.

## 18. Stage 1 Failure Mechanism

Stage 1 trained a tiny transformer on short sequences. It performed well near training length but failed on much longer exactly-one-positive sequences.

The observed pattern was:

- target attention mass decreased with length
- positive logits decreased with length
- max-pooled non-target contribution became more negative with length

A reduced view is:

```math
\operatorname{logit}(n)
\approx
\operatorname{bias}
+
\operatorname{target\ signal}(n)
-
\operatorname{non\ target\ interference}(n).
```

Variables used here:

- $\operatorname{logit}(n)$ is the final classifier logit at sequence length $n$.
- $\operatorname{bias}$ is a length-independent baseline term.
- $\operatorname{target\ signal}(n)$ is the positive contribution from detecting the target.
- $\operatorname{non\ target\ interference}(n)$ is the negative contribution that grows or changes with non-target tokens.

Interpretation:

The model did not learn a clean length-invariant algorithm. It learned mechanisms that worked at short length but degraded as the number of non-target tokens increased.

## 19. Simplified Attention Model

The professor's simplified model isolates the attention part.

Assume the last query sees one target score and many identical non-target scores:

```math
S_n
=
(a,b,b,\ldots,b).
```

Variables used here:

- $S_n$ is the last-query attention score row for sequence length $n$.
- $a$ is the score assigned to the target key.
- $b$ is the score assigned to each non-target key.

Apply inverse temperature $\alpha$:

```math
\operatorname{softmax}(\alpha S_n).
```

Variables used here:

- $\alpha$ is the inverse temperature or score scaling factor.
- $S_n$ is the score row before scaling.
- $\operatorname{softmax}$ converts scaled scores into attention weights.

The target attention mass is:

```math
p_t(n)
=
\frac{e^{\alpha(a-b)}}
{e^{\alpha(a-b)} + (n-1)}.
```

Variables used here:

- $p_t(n)$ is the target attention mass.
- $\alpha$ is the inverse temperature.
- $a-b$ is the target-vs-non-target score margin.
- $n-1$ is the number of non-target keys.

Let:

```math
\Delta = a-b.
```

Variables used here:

- $\Delta$ is the target-vs-non-target score margin.
- $a$ is the target score.
- $b$ is the non-target score.

Then:

```math
p_t(n)
=
\frac{e^{\alpha\Delta}}
{e^{\alpha\Delta} + (n-1)}.
```

Variables used here:

- $p_t(n)$ is target attention mass.
- $\alpha\Delta$ is the scaled score margin.
- $n-1$ is the number of non-target keys.

The key condition is:

```math
\alpha\Delta - \log n \to +\infty.
```

Variables used here:

- $\alpha\Delta$ is the scaled target advantage.
- $\log n$ represents the asymptotic growth of the number of non-target competitors.
- $\to +\infty$ means the difference grows without bound as $n$ increases.

Interpretation:

The scaled target margin must grow faster than the logarithm of the number of competing non-target tokens.

## 20. Why $\alpha = \log n$ Depends On $\Delta$

If:

```math
\alpha = \log n,
```

Variables used here:

- $\alpha$ is set equal to the logarithm of sequence length.
- $n$ is sequence length.

then:

```math
p_t(n)
=
\frac{n^\Delta}
{n^\Delta + n - 1}.
```

Variables used here:

- $p_t(n)$ is target attention mass.
- $n^\Delta$ comes from $e^{(\log n)\Delta}$.
- $\Delta$ is the target-vs-non-target score margin.

The limit depends on $\Delta$:

```math
\Delta > 1
\Rightarrow
p_t(n) \to 1.
```

```math
\Delta = 1
\Rightarrow
p_t(n) \to \frac{1}{2}.
```

```math
0 < \Delta < 1
\Rightarrow
p_t(n) \to 0.
```

Interpretation:

A log-length temperature is not enough by itself. It only works if the target score margin is large enough.

## 21. Stage 2B Interventions

Stage 2B tries to prevent length degradation by modifying attention scores.

### Global Log-Temperature

The global log-temperature intervention scales all attention scores:

```math
\tilde{s}_{ij}
=
\alpha s_{ij}.
```

Variables used here:

- $\tilde{s}_{ij}$ is the modified attention score.
- $s_{ij}$ is the raw attention score from query position $i$ to key position $j$.
- $\alpha$ is the global score scaling factor.

Interpretation:

This increases score gaps globally. In the simplified model, it helps only if it makes $\alpha\Delta$ large enough compared with $\log n$.

### Target-Key Log-Bias

The target-key bias intervention adds a key-specific term:

```math
\tilde{s}_{ij}
=
s_{ij}
+
\beta r_j.
```

Variables used here:

- $\tilde{s}_{ij}$ is the modified attention score.
- $s_{ij}$ is the raw attention score.
- $\beta$ is the additive bias scale.
- $r_j$ is a learned target-likeness score for key position $j$.

Interpretation:

This is more targeted than global scaling. It can increase the target-vs-non-target margin directly if $r_t > r_u$.

## 22. Connecting The Documents

Read the project documents in this order:

1. `STAGE1_NUMERICAL_ANALYSIS.md`
2. `THEORETICAL_MODEL.md`
3. `SIMPLIFIED_LENGTH_AWARE_ATTENTION_MODEL.md`
4. `STAGE2B_LENGTH_AWARE_ATTENTION_ANALYSIS.md`

The connection is:

```text
Stage 1 observed the failure.
THEORETICAL_MODEL.md summarized the failure with a reduced formula.
SIMPLIFIED_LENGTH_AWARE_ATTENTION_MODEL.md isolates the attention dilution condition.
Stage 2B tests whether length-aware score modifications can satisfy that condition.
```

## 23. Key Concepts To Remember

The most important ideas are:

- A logit is the raw classifier output.
- Positive logits predict target present.
- Attention weights are normalized scores.
- The softmax denominator grows with sequence length.
- A fixed target margin is not enough for infinite length.
- Max pooling can introduce length-growing non-target interference.
- Stage 2B tries to increase the effective target margin as length grows.

The main mathematical condition to remember is:

```math
\alpha\Delta - \log n \to +\infty.
```

Variables used here:

- $\alpha$ is the score scaling factor.
- $\Delta$ is the target-vs-non-target score margin.
- $n$ is sequence length.
- The expression says the scaled target margin must grow faster than $\log n$.

This condition means:

```text
The scaled target advantage must beat the growth in the number of non-target tokens.
```