# Reduced Theoretical Model for Stage 1 Length Failure

## Objective

This document defines a reduced mathematical model for the Stage 1 transformer failure on long exactly-one positive sequences.

The goal is not to reproduce every internal activation of the trained transformer. The goal is to write a small formula that captures the main empirical pattern observed in `ANALYSIS.md`:

- target attention mass decreases as sequence length grows
- non-target interference increases with sequence length
- the final positive logit eventually crosses below zero

## High-Level Decomposition

We model the final classifier logit as the sum of three terms:

```math
\mathrm{final\_logit}(\mathrm{sequence\_length})
\approx
\mathrm{classifier\_bias}
+
\mathrm{target\_signal\_contribution}(\mathrm{sequence\_length})
-
\mathrm{non\_target\_interference\_contribution}(\mathrm{sequence\_length})
```

The intended interpretation is:

```math
\mathrm{target\_signal\_contribution}(\mathrm{sequence\_length})
\text{ decreases or saturates as length grows}
```

while:

```math
\mathrm{non\_target\_interference\_contribution}(\mathrm{sequence\_length})
\text{ increases as length grows}
```

The Stage 1 model fails when:

```math
\mathrm{final\_logit}(\mathrm{sequence\_length}) < 0
```

because the binary decision rule is:

```math
\mathrm{prediction}
=
\begin{cases}
\mathrm{positive}, & \mathrm{final\_logit} \ge 0 \\
\mathrm{negative}, & \mathrm{final\_logit} < 0
\end{cases}
```

## Target Attention Mass

For one query position, define:

```math
\mathrm{target\_attention\_score}
```

as the attention score assigned to the target key, and define:

```math
\mathrm{expected\_non\_target\_exp\_score}
```

as the average exponentiated attention score assigned to non-target keys.

The approximate attention mass on the target key is:

```math
\mathrm{target\_attention\_mass}(\mathrm{sequence\_length})
=
\frac{
\exp(\mathrm{target\_attention\_score})
}{
\exp(\mathrm{target\_attention\_score})
+
(\mathrm{sequence\_length} - 1)
\mathrm{expected\_non\_target\_exp\_score}
}
```

This formula captures the softmax denominator effect. The target numerator may remain large, but the denominator grows as more non-target keys are added.

## Score Margin Form

Define the fixed target score margin:

```math
\mathrm{target\_score\_margin}
=
\mathrm{target\_attention\_score}
-
\log(\mathrm{expected\_non\_target\_exp\_score})
```

Then the target attention mass can be rewritten as:

```math
\mathrm{target\_attention\_mass}(\mathrm{sequence\_length})
=
\frac{
1
}{
1
+
(\mathrm{sequence\_length} - 1)
\exp(-\mathrm{target\_score\_margin})
}
```

This is the core length-scaling formula. If `target_score_margin` is fixed, then `target_attention_mass` decreases as `sequence_length` increases.

## Required Margin for Length Generalization

To keep the target attention mass roughly constant as length grows, the target score margin must grow with length.

A rough condition is:

```math
\mathrm{target\_score\_margin}
\gtrsim
\log(\mathrm{sequence\_length})
```

For example:

```math
\log(10) \approx 2.3
```

while:

```math
\log(1000) \approx 6.9
```

Therefore, a margin that is sufficient for length 10 can be insufficient for length 1000.

This is a plausible reason why the Stage 1 transformer succeeds on length-10 training examples but fails on much longer positive examples. Length-10 training does not strongly pressure the model to learn the much larger margin needed for long-sequence extrapolation.

## Final Logit Model

Attention dilution alone does not fully describe the final classifier output, because the Stage 1 model uses max pooling before the classifier.

The reduced logit model is:

```math
\mathrm{final\_logit}(\mathrm{sequence\_length})
\approx
\mathrm{classifier\_bias}
+
\mathrm{target\_signal\_strength}
\cdot
\mathrm{target\_attention\_mass}(\mathrm{sequence\_length})
-
\mathrm{non\_target\_interference\_strength}
\cdot
\mathrm{non\_target\_interference\_growth}(\mathrm{sequence\_length})
```

The terms are:

```math
\mathrm{classifier\_bias}
```

the bias term of the final linear classifier;

```math
\mathrm{target\_signal\_strength}
```

the conversion strength from target attention mass into positive classifier evidence;

```math
\mathrm{non\_target\_interference\_strength}
```

the strength with which non-target positions push the classifier logit downward;

```math
\mathrm{non\_target\_interference\_growth}(\mathrm{sequence\_length})
```

the length-dependent growth of non-target interference.

## Candidate Non-Target Interference Functions

The empirical analysis showed that target-sourced max-pool contribution stays relatively stable at long lengths, while non-target-sourced contribution becomes increasingly negative.

One simple candidate is logarithmic growth:

```math
\mathrm{non\_target\_interference\_growth}(\mathrm{sequence\_length})
=
\log(\mathrm{sequence\_length})
```

Another candidate comes from the intuition that max pooling over many non-target positions creates an extreme-value effect:

```math
\mathrm{non\_target\_interference\_growth}(\mathrm{sequence\_length})
=
\sqrt{2\log(\mathrm{sequence\_length})}
```

Both should be treated as candidate reduced models. The next step is to fit each candidate to the empirical length sweep and compare predicted logits against observed logits.

## Complete Reduced Formula

The complete candidate model is:

```math
\mathrm{target\_attention\_mass}(\mathrm{sequence\_length})
=
\frac{
1
}{
1
+
(\mathrm{sequence\_length} - 1)
\exp(-\mathrm{target\_score\_margin})
}
```

and:

```math
\mathrm{final\_logit}(\mathrm{sequence\_length})
\approx
\mathrm{classifier\_bias}
+
\mathrm{target\_signal\_strength}
\cdot
\mathrm{target\_attention\_mass}(\mathrm{sequence\_length})
-
\mathrm{non\_target\_interference\_strength}
\cdot
\mathrm{non\_target\_interference\_growth}(\mathrm{sequence\_length})
```

The main fitted parameters are:

```math
\mathrm{target\_score\_margin}
```

```math
\mathrm{target\_signal\_strength}
```

```math
\mathrm{non\_target\_interference\_strength}
```

```math
\mathrm{classifier\_bias}
```

The main model-selection choice is:

```math
\mathrm{non\_target\_interference\_growth}(\mathrm{sequence\_length})
\in
\left\{
\log(\mathrm{sequence\_length}),
\sqrt{2\log(\mathrm{sequence\_length})}
\right\}
```

## Interpretation

This reduced model explains the empirical failure as:

```math
\mathrm{sequence\_length} \uparrow
\quad\Rightarrow\quad
\mathrm{softmax\_denominator} \uparrow
\quad\Rightarrow\quad
\mathrm{target\_attention\_mass} \downarrow
```

and:

```math
\mathrm{sequence\_length} \uparrow
\quad\Rightarrow\quad
\mathrm{non\_target\_interference\_growth} \uparrow
\quad\Rightarrow\quad
\mathrm{negative\_classifier\_contribution} \uparrow
```

Together:

```math
\mathrm{final\_logit}(\mathrm{sequence\_length}) \downarrow
```

The Stage 1 transformer therefore appears to learn a finite-margin target-detection mechanism rather than a true length-invariant existential algorithm.

## Empirical Fit Results

The formula was fit to the extended Stage 1 length sweep in:

```text
runs/stage1_transformer_maxpool2/numerical_analysis/theoretical_fit
```

The evaluated lengths were:

```text
10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900,
950, 1000, 1050, 1100, 1200, 1500, 2000, 3000, 5000, 10000
```

### Attention Fit

The target attention curve is well explained by the fixed-margin softmax dilution formula:

```math
\mathrm{target\_attention\_mass}(\mathrm{sequence\_length})
=
\frac{
1
}{
1
+
(\mathrm{sequence\_length} - 1)
\exp(-\mathrm{target\_score\_margin})
}
```

Fit result:

| Quantity | Value |
|---|---:|
| target score margin | 5.9855 |
| R-squared | 0.9586 |
| mean absolute error | 0.0360 |
| root mean squared error | 0.0472 |

<img src="figures/target_attention_fit.png" alt="Target attention formula fit" width="560">

This supports the fixed-margin interpretation. The learned effective margin is enough for short sequences, but it is smaller than the rough margin needed at long lengths:

```math
\log(1000) \approx 6.91
```

```math
\log(10000) \approx 9.21
```

### Final Logit Fit

The best reduced logit formula was:

```math
\mathrm{final\_logit}(\mathrm{sequence\_length})
\approx
7.2765
+
6.8618
\cdot
\mathrm{observed\_target\_attention\_mass}(\mathrm{sequence\_length})
-
1.4136
\cdot
\log(\mathrm{sequence\_length})
```

Fit result:

| Quantity | Value |
|---|---:|
| R-squared | 0.9846 |
| mean absolute error | 0.4099 |
| root mean squared error | 0.4861 |

<img src="figures/positive_logit_fit.png" alt="Positive logit formula fit" width="560">

This means that target attention decay alone is not the full explanation. The final logit is better explained by combining positive target evidence with a length-growing non-target penalty.

### Max-Pool Contribution Fit

The measured non-target-sourced max-pool contribution is also well fit by a logarithmic length penalty:

```math
\mathrm{non\_target\_sourced\_contribution}(\mathrm{sequence\_length})
\approx
13.2757
-
2.2212
\cdot
\log(\mathrm{sequence\_length})
```

Fit result:

| Model | R-squared | RMSE |
|---|---:|---:|
| non-target constant | 0.0000 | 3.7056 |
| non-target log(sequence_length) | 0.9579 | 0.7606 |
| non-target sqrt(2 log(sequence_length)) | 0.9174 | 1.0648 |

<img src="figures/maxpool_non_target_contribution_fit.png" alt="Non-target max-pool contribution fit" width="560">

This supports the interpretation that the log-length penalty in the final-logit formula corresponds to a real max-pool interference effect, not just an arbitrary curve fit.

The target-sourced contribution is comparatively stable but not exactly constant:

| Model | RMSE |
|---|---:|
| target constant | 0.4395 |

<img src="figures/maxpool_target_contribution_fit.png" alt="Target max-pool contribution fit" width="560">

Therefore, the safer statement is:

```text
Target-sourced contribution changes much less than non-target-sourced contribution,
but it should not be treated as perfectly constant.
```

## Current Conclusion

The extended fit supports the reduced explanation:

- fixed target score margin
- softmax denominator growth
- length-growing non-target interference

More concretely:

```math
\mathrm{target\_attention\_mass}
\text{ is explained by fixed-margin softmax dilution}
```

and:

```math
\mathrm{final\_logit}
\text{ is explained by target evidence minus a logarithmic non-target max-pool penalty}
```

This is a close quantitative reproduction of the empirical Stage 1 failure curve, although it remains a reduced model rather than a full exact model of every transformer activation.
