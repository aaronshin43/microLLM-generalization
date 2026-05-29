# Experiment Notes

## Stage 0: Max-Pooling Baseline

The non-transformer max-pooling baseline generalizes perfectly across the tested length sweep. This confirms that the task itself is length-generalizable when the architecture directly matches the existential rule.

The diagnostic slice evaluation also remains perfect across all tested lengths. The baseline reaches 1.0000 accuracy for zero-target negatives, exactly-one positives, multi-target positives, and target-position slices near the beginning, middle, and end of the sequence.

This is an important control result: the later transformer failure is not caused by the data generator, the evaluation slices, or the intrinsic difficulty of the synthetic task. A simple permutation-invariant max-pooling detector solves all diagnostic cases cleanly.

## Stage 1: Minimal Transformer

Configuration:

- 1 transformer encoder layer
- 1 attention head
- `d_model = 64`
- no positional encoding
- max pooling
- trained only on length 10

Main finding:

The transformer learns the task at length 10 and extrapolates well up to length 850, but it does not learn a truly length-invariant existential algorithm. Positive accuracy begins to degrade around length 900 and collapses by length 1000-1100, while negative accuracy remains perfect.

Primary length sweep:

| Length | Positive Accuracy | Negative Accuracy |
|---:|---:|---:|
| 850 | 1.0000 | 1.0000 |
| 900 | 0.9976 | 1.0000 |
| 950 | 0.8968 | 1.0000 |
| 1000 | 0.4234 | 1.0000 |
| 1100 | 0.0006 | 1.0000 |

Diagnostic slices show that the failure is specific to sparse exactly-one positives. Multi-target positives remain easy even at long lengths, and target position does not substantially change the failure pattern.

| Slice at Length 1000 | Accuracy |
|---|---:|
| negative, zero target | 1.0000 |
| positive, exactly one target | 0.4320 |
| positive, 3 targets | 1.0000 |
| positive, 10 targets | 1.0000 |
| positive, 1% target density | 1.0000 |
| positive, target near beginning | 0.4270 |
| positive, target near middle | 0.4240 |
| positive, target near end | 0.4025 |

Attention diagnostics support a length-dependent signal dilution interpretation. As sequence length grows, attention becomes more diffuse and the average attention mass assigned to the target token decreases.

Selected positive-example attention diagnostics:

| Length | Avg Positive Logit | Target Attention Mean | Attention Entropy Mean |
|---:|---:|---:|---:|
| 10 | 9.40 | 0.879 | 0.45 |
| 500 | 1.85 | 0.438 | 3.87 |
| 900 | 0.43 | 0.339 | 4.80 |
| 1000 | -0.03 | 0.306 | 5.06 |
| 1100 | -0.58 | 0.298 | 5.18 |

Interpretation:

The model appears to learn a useful but length-fragile target-detection mechanism. It can detect targets when the signal is sufficiently strong, but in the exactly-one sparse case the positive logit margin shrinks with length and eventually crosses the decision boundary. This is evidence against true infinite-length generalization for this minimal transformer configuration.

## Stage 2A: Multi-Length Transformer Training

Configuration:

- same transformer architecture as Stage 1
- 1 transformer encoder layer
- 1 attention head
- `d_model = 64`
- no positional encoding
- max pooling
- trained on lengths `[10, 20, 50, 100]`

Main finding:

Multi-length training greatly improves extrapolation, but it still does not produce true infinite-length generalization. The model remains perfect through length 8000, but sparse exactly-one positives collapse at length 10000.

Primary length sweep:

| Length | Positive Accuracy | Negative Accuracy |
|---:|---:|---:|
| 1000 | 1.0000 | 1.0000 |
| 1500 | 1.0000 | 1.0000 |
| 3000 | 1.0000 | 1.0000 |
| 5000 | 1.0000 | 1.0000 |
| 8000 | 1.0000 | 1.0000 |
| 10000 | 0.0000 | 1.0000 |

Diagnostic slices show that the length-10000 failure is specific to sparse exactly-one positives. Multi-target positives remain correct.

| Slice at Length 10000 | Accuracy |
|---|---:|
| negative, zero target | 1.0000 |
| positive, exactly one target | 0.0000 |
| positive, 3 targets | 1.0000 |
| positive, 10 targets | 1.0000 |
| positive, 1% target density | 1.0000 |
| positive, target near beginning | 0.0000 |
| positive, target near middle | 0.0000 |
| positive, target near end | 0.0000 |

Attention diagnostics make the failure mechanism clear. For exactly-one positives, the target attention mass and logit margin both decline as length increases. At length 8000 the model is still correct, but the average logit is only slightly positive. At length 10000 it becomes negative.

Selected exactly-one positive attention diagnostics:

| Length | Avg Positive Logit | Target Attention Mean | Target Attention Max | Attention Entropy Mean |
|---:|---:|---:|---:|---:|
| 10 | 10.63 | 0.725 | 0.996 | 0.62 |
| 1000 | 8.17 | 0.395 | 0.943 | 4.24 |
| 3000 | 2.94 | 0.259 | 0.846 | 5.85 |
| 5000 | 1.91 | 0.204 | 0.765 | 6.61 |
| 8000 | 0.39 | 0.161 | 0.677 | 7.29 |
| 10000 | -0.58 | 0.142 | 0.626 | 7.62 |

Interpretation:

Stage 2A shifts the Stage 1 failure point from around length 1000 to between lengths 8000 and 10000. This indicates that the Stage 1 failure was strongly influenced by the single-length training distribution. However, the Stage 2A model still appears to rely on a length-sensitive confidence margin rather than a fully length-invariant boolean OR rule. Multiple target tokens provide enough evidence even at length 10000, but one target token no longer produces a positive logit.
