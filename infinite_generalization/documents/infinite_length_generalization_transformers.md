# Infinite-Length Generalization for Transformers

## Research Motivation

An undergraduate student is going to start working on a research project that can be thought of as **infinite-length generalization for transformers**. The idea may only work for simple tasks, but it is worth exploring.

As a starting example, consider a task that requires no positional information. For instance, the model receives a sequence of tokens and produces a binary output indicating whether a certain token is present anywhere in the input.

Suppose we train a PyTorch transformer model to do this with essentially 100% accuracy when the context window is 10 tokens. The question is:

> Can we take exactly the same model, give it a larger context window, and expect it to work the same way?

This document summarizes the main practical and theoretical issues involved in that experiment.

---

## Short Answer

Yes, this is a very reasonable experiment. For the specific task “does token X occur anywhere?”, there is no fundamental PyTorch obstacle to evaluating the same trained model on longer sequences.

However, whether the model actually works on longer sequences depends on architectural details and on whether it has learned a genuinely length-invariant algorithm.

A useful guiding principle is:

> If the model is genuinely length-agnostic, uses no fixed-size positional embeddings, and reduces the sequence dimension with something like mean pooling, max pooling, sum pooling, or attention pooling rather than flattening, then it can usually be evaluated on longer sequences without changing the learned weights.

But 100% accuracy at length 10 does **not** imply correctness at length 20, 100, or 1000, even for a simple task.

---

## 1. PyTorch Is Not the Main Barrier

A typical PyTorch transformer layer does not intrinsically care whether the sequence length is 10 or 100. For example, a multi-head attention module is defined over query, key, and value tensors with a sequence-length dimension. With `batch_first=True`, the expected shape is:

```python
(batch_size, sequence_length, embedding_dimension)
```

So the same module can often be called with different sequence lengths:

```python
# trained on x.shape == (batch, 10)
logits = model(x)

# later evaluated on x_long.shape == (batch, 50)
logits_long = model(x_long)
```

The learned matrices in attention and MLP blocks usually do not depend on the context length. The context length appears in runtime tensors, especially the attention score matrix, rather than in the parameter shapes.

---

## 2. Architectural Traps

The model will **not** be length-agnostic if it contains a layer or buffer whose shape depends on the training length.

### Fixed Learned Absolute Positional Embeddings

For example:

```python
self.pos_embedding = nn.Parameter(torch.randn(10, d_model))
```

This model cannot directly process length 50, because it only has 10 position vectors. The positional embedding table would need to be extended, interpolated, replaced, or removed.

For the initial “token present anywhere” task, it is best to omit positional encodings entirely.

### Flattening the Whole Sequence Before Classification

This is not length-agnostic:

```python
x = x.reshape(batch_size, 10 * d_model)
logits = self.classifier(x)
```

The classifier is hard-coded for length 10.

Instead, use a length-independent reduction across the sequence dimension:

```python
x = x.mean(dim=1)       # mean pooling over tokens
logits = self.classifier(x)
```

or:

```python
x = x.max(dim=1).values # max pooling over tokens
logits = self.classifier(x)
```

A learned `[CLS]` token is another possibility, although it has its own issues for length extrapolation.

### Hard-Coded Attention Masks

If the model uses an attention mask or padding mask, make sure the mask is generated dynamically from the current sequence length.

For example, a fixed mask of shape `(10, 10)` will not work for length 50 unless it is regenerated or extended.

### Manually Stored Buffers of Length 10

Examples include:

- a precomputed positional encoding buffer of shape `(10, d_model)`;
- a fixed causal mask of shape `(10, 10)`;
- a learned or stored sequence-level parameter indexed by position.

These will either break directly or silently impose length-specific behavior.

---

## 3. The Main Theoretical Issue: The Learned Algorithm May Not Extrapolate

For the task “is token T present anywhere?”, there exists a simple length-generalizing solution:

```python
present = max_i indicator(x_i == T)
```

That algorithm works at every sequence length.

But a model trained only on length 10 may learn a length-specific shortcut.

### Mean Pooling Example

Suppose each token is mapped to a feature that is 1 for the target token and 0 otherwise. Mean pooling gives:

```text
number of target tokens / sequence length
```

At length 10, one occurrence gives:

```text
1 / 10 = 0.1
```

At length 100, one occurrence gives:

```text
1 / 100 = 0.01
```

A classifier trained on length 10 might learn:

```text
positive if pooled feature > 0.05
```

That rule works perfectly at length 10, but fails at length 100 when there is exactly one target token.

This is one of the most important conceptual barriers. The architecture may accept longer sequences, but the internal numerical scale may change with length.

---

## 4. Pooling Choices Matter

For the first experiment, it would be useful to compare several ways of reducing the sequence dimension.

| Pooling method | Likely length behavior |
|---|---|
| Mean pooling | May fail because signal is diluted as length grows. |
| Sum pooling | Preserves a single-token signal, but negative examples may drift with length. |
| Max pooling | Best match to existential “is token present?” tasks. |
| `[CLS]` attention pooling | Can work, but may fail if attention remains diffuse. |

For this particular task, max pooling is especially natural because the target property is existential:

```text
Does there exist a position containing the target token?
```

---

## 5. Attention Can Also Dilute the Signal

Self-attention uses a softmax over positions. If a query attends roughly uniformly to all tokens, then the contribution of one special token is approximately:

```text
1 / sequence_length
```

So a target token that is easy to detect at length 10 may become much weaker at length 100.

The model can overcome this by learning a sharp attention pattern, such as:

```text
attend almost entirely to the target token if it exists
```

But training at length 10 does not guarantee that the model has learned this length-general algorithm.

So the research question is not merely:

> Can the model run on longer sequences?

The deeper question is:

> Did the model learn a length-invariant computation?

---

## 6. Computational Barriers

Even if the model is architecturally length-agnostic, standard full attention has quadratic cost in sequence length.

The attention score matrix has approximate shape:

```text
batch_size × num_heads × sequence_length × sequence_length
```

So moving from length 10 to length 100 increases the attention matrix size by a factor of 100. Moving from length 10 to length 1000 increases it by a factor of 10,000.

For early toy experiments, this is probably not a problem. But it matters if the student eventually tries very long sequences.

---

## 7. A Simple Non-Transformer Baseline

The “token present anywhere” task can be solved by a very simple permutation-invariant architecture:

```text
token embedding → per-token detector → max over sequence → classifier
```

This is useful as a baseline. Then the research question becomes sharper:

> Can a small transformer discover an algorithm equivalent to this simple invariant detector, and does it extrapolate beyond the training length?

A possible baseline architecture is:

```text
Embedding
Linear/ReLU per token
max over sequence
Linear classifier
```

If this baseline generalizes perfectly to length 1000 but the transformer does not, that is an interesting result rather than a failure.

---

## 8. Suggested First Experimental Design

Train only on sequences of length 10. Then evaluate on:

```text
10, 20, 50, 100, 200, 500, 1000
```

Keep the positive/negative distribution controlled. In particular, test separate cases:

```text
positive with exactly 1 target token
positive with several target tokens
negative with no target token
target token near beginning
target token near middle
target token near end
```

A good starting architecture would be:

```text
token embedding
no positional encoding
1–2 transformer encoder layers
sequence pooling
binary classifier
```

Then compare:

```text
mean pooling
sum pooling
max pooling
CLS token
```

Prediction:

> Max pooling should extrapolate best for this task. Mean pooling may look perfect at length 10 but fail badly as length increases. A `[CLS]`-style transformer may or may not extrapolate, depending on whether it learns sharp attention to the target token.

---

## 9. Bottom Line

There is no deep PyTorch barrier to expanding the context window, provided the model does not contain length-specific components.

The main barriers are:

1. fixed positional embeddings or fixed masks;
2. classifiers that flatten a length-10 sequence;
3. quadratic attention cost;
4. length-dependent numerical scaling, especially from mean pooling or softmax attention;
5. the fact that perfect training/test accuracy at length 10 does not prove that the model learned the intended length-general algorithm.

For this project, the last point is the interesting research space.

