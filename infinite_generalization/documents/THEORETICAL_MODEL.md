# Reduced Theoretical Model for Stage 1 Length Failure

## Objective

This document defines a reduced mathematical model for the Stage 1 transformer failure on long exactly-one positive sequences.

The goal is not to reproduce every internal activation of the trained transformer. The goal is to write a small formula that captures the main empirical pattern observed in `ANALYSIS.md`:

- target attention mass decreases as sequence length grows
- non-target interference increases with sequence length
- the final positive logit eventually crosses below zero

## High-Level Decomposition

We model the final classifier logit as the sum of three terms:

$$
\operatorname{final\_logit}(\operatorname{sequence\_length})
\approx
\operatorname{classifier\_bias}
+
\operatorname{target\_signal\_contribution}(\operatorname{sequence\_length})
-
\operatorname{non\_target\_interference\_contribution}(\operatorname{sequence\_length})
$$

The intended interpretation is:

$$
\operatorname{target\_signal\_contribution}(\operatorname{sequence\_length})
\text{ decreases or saturates as length grows}
$$

while:

$$
\operatorname{non\_target\_interference\_contribution}(\operatorname{sequence\_length})
\text{ increases as length grows}
$$

The Stage 1 model fails when:

$$
\operatorname{final\_logit}(\operatorname{sequence\_length}) < 0
$$

because the binary decision rule is:

$$
\operatorname{prediction}
=
\begin{cases}
\operatorname{positive}, & \operatorname{final\_logit} \ge 0 \\
\operatorname{negative}, & \operatorname{final\_logit} < 0
\end{cases}
$$

## Target Attention Mass

For one query position, define:

$$
\operatorname{target\_attention\_score}
$$

as the attention score assigned to the target key, and define:

$$
\operatorname{expected\_non\_target\_exp\_score}
$$

as the average exponentiated attention score assigned to non-target keys.

The approximate attention mass on the target key is:

$$
\operatorname{target\_attention\_mass}(\operatorname{sequence\_length})
=
\frac{
\exp(\operatorname{target\_attention\_score})
}{
\exp(\operatorname{target\_attention\_score})
+
(\operatorname{sequence\_length} - 1)
\operatorname{expected\_non\_target\_exp\_score}
}
$$

This formula captures the softmax denominator effect. The target numerator may remain large, but the denominator grows as more non-target keys are added.

## Score Margin Form

Define the fixed target score margin:

$$
\operatorname{target\_score\_margin}
=
\operatorname{target\_attention\_score}
-
\log(\operatorname{expected\_non\_target\_exp\_score})
$$

Then the target attention mass can be rewritten as:

$$
\operatorname{target\_attention\_mass}(\operatorname{sequence\_length})
=
\frac{
1
}{
1
+
(\operatorname{sequence\_length} - 1)
\exp(-\operatorname{target\_score\_margin})
}
$$

This is the core length-scaling formula. If `target_score_margin` is fixed, then `target_attention_mass` decreases as `sequence_length` increases.

## Required Margin for Length Generalization

To keep the target attention mass roughly constant as length grows, the target score margin must grow with length.

A rough condition is:

$$
\operatorname{target\_score\_margin}
\gtrsim
\log(\operatorname{sequence\_length})
$$

For example:

$$
\log(10) \approx 2.3
$$

while:

$$
\log(1000) \approx 6.9
$$

Therefore, a margin that is sufficient for length 10 can be insufficient for length 1000.

This is a plausible reason why the Stage 1 transformer succeeds on length-10 training examples but fails on much longer positive examples. Length-10 training does not strongly pressure the model to learn the much larger margin needed for long-sequence extrapolation.

## Final Logit Model

Attention dilution alone does not fully describe the final classifier output, because the Stage 1 model uses max pooling before the classifier.

The reduced logit model is:

$$
\operatorname{final\_logit}(\operatorname{sequence\_length})
\approx
\operatorname{classifier\_bias}
+
\operatorname{target\_signal\_strength}
\cdot
\operatorname{target\_attention\_mass}(\operatorname{sequence\_length})
-
\operatorname{non\_target\_interference\_strength}
\cdot
\operatorname{non\_target\_interference\_growth}(\operatorname{sequence\_length})
$$

The terms are:

$$
\operatorname{classifier\_bias}
$$

the bias term of the final linear classifier;

$$
\operatorname{target\_signal\_strength}
$$

the conversion strength from target attention mass into positive classifier evidence;

$$
\operatorname{non\_target\_interference\_strength}
$$

the strength with which non-target positions push the classifier logit downward;

$$
\operatorname{non\_target\_interference\_growth}(\operatorname{sequence\_length})
$$

the length-dependent growth of non-target interference.

## Candidate Non-Target Interference Functions

The empirical analysis showed that target-sourced max-pool contribution stays relatively stable at long lengths, while non-target-sourced contribution becomes increasingly negative.

One simple candidate is logarithmic growth:

$$
\operatorname{non\_target\_interference\_growth}(\operatorname{sequence\_length})
=
\log(\operatorname{sequence\_length})
$$

Another candidate comes from the intuition that max pooling over many non-target positions creates an extreme-value effect:

$$
\operatorname{non\_target\_interference\_growth}(\operatorname{sequence\_length})
=
\sqrt{2\log(\operatorname{sequence\_length})}
$$

Both should be treated as candidate reduced models. The next step is to fit each candidate to the empirical length sweep and compare predicted logits against observed logits.

## Complete Reduced Formula

The complete candidate model is:

$$
\operatorname{target\_attention\_mass}(\operatorname{sequence\_length})
=
\frac{
1
}{
1
+
(\operatorname{sequence\_length} - 1)
\exp(-\operatorname{target\_score\_margin})
}
$$

and:

$$
\operatorname{final\_logit}(\operatorname{sequence\_length})
\approx
\operatorname{classifier\_bias}
+
\operatorname{target\_signal\_strength}
\cdot
\operatorname{target\_attention\_mass}(\operatorname{sequence\_length})
-
\operatorname{non\_target\_interference\_strength}
\cdot
\operatorname{non\_target\_interference\_growth}(\operatorname{sequence\_length})
$$

The main fitted parameters are:

$$
\operatorname{target\_score\_margin}
$$

$$
\operatorname{target\_signal\_strength}
$$

$$
\operatorname{non\_target\_interference\_strength}
$$

$$
\operatorname{classifier\_bias}
$$

The main model-selection choice is:

$$
\operatorname{non\_target\_interference\_growth}(\operatorname{sequence\_length})
\in
\left\{
\log(\operatorname{sequence\_length}),
\sqrt{2\log(\operatorname{sequence\_length})}
\right\}
$$

## Interpretation

This reduced model explains the empirical failure as:

$$
\operatorname{sequence\_length} \uparrow
\quad\Rightarrow\quad
\operatorname{softmax\_denominator} \uparrow
\quad\Rightarrow\quad
\operatorname{target\_attention\_mass} \downarrow
$$

and:

$$
\operatorname{sequence\_length} \uparrow
\quad\Rightarrow\quad
\operatorname{non\_target\_interference\_growth} \uparrow
\quad\Rightarrow\quad
\operatorname{negative\_classifier\_contribution} \uparrow
$$

Together:

$$
\operatorname{final\_logit}(\operatorname{sequence\_length}) \downarrow
$$

The Stage 1 transformer therefore appears to learn a finite-margin target-detection mechanism rather than a true length-invariant existential algorithm.

## Next Step

Fit the reduced formula to the empirical Stage 1 length sweep:

$$
\operatorname{observed\_target\_attention\_mass}(\operatorname{sequence\_length})
$$

$$
\operatorname{observed\_maxpool\_source\_contributions}(\operatorname{sequence\_length})
$$

$$
\operatorname{observed\_final\_logit}(\operatorname{sequence\_length})
$$

The fit should answer whether the observed failure can be quantitatively explained by:

- fixed target score margin
- softmax denominator growth
- length-growing non-target interference
