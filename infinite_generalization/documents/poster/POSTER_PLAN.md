# Poster Plan

## Format and Audience

- Size: 40 x 30 inches
- Orientation: landscape
- Grid: four columns with two primary content rows
- Audience: a general academic audience, including readers without prior
  attention or length-generalization knowledge
- Style: minimal text, calm tone, and accessible language
- Reading order: problem, intervention, and evidence
- Communication layers: the headline and visual provide the intuition, concise
  prose states the claim, and an optional equation gives the formal result

## Core Claims

1. A target can outrank every non-target individually yet receive a shrinking
   share of attention as the sequence grows.
2. Sufficiently strong logarithmic score scaling prevents this dilution in the
   reduced attention model.

Supporting message:

> A long finite test can still hide eventual collapse.

Results framing:

> Same training-length accuracy. Sharply different long-length behavior.

## Header

### Main Title

> **Attention Beyond the Training Length**

### Subtitle

> **How Score Scaling Controls Length Generalization in a Reduced Binary
> Attention Classifier**

### Author and Affiliation

> **Aaron Shin**  
> Deep Network Understanding Lab, Dickinson College

### Visual Specification

- Set the main title as the largest text on the poster.
- Place the subtitle directly below it at a clearly subordinate size.
- Keep the author and affiliation visually separate from the title block.
- Do not add another central-claim sentence to the header; the subtitle and the
  Motivation block already establish the topic and scope.

### Status

- Main title: confirmed
- Subtitle: confirmed
- Author and affiliation: confirmed

## Overall Storyboard

1. **Motivation and research question:** state the short-to-long generalization
   problem and ask when length-aware attention prevents target dilution.
2. **Why fixed attention dilutes with length:** introduce the task and contrast
   an individual target advantage with the growing aggregate non-target weight.
3. **Model and score scaling:** introduce constant, logarithmic, and learned
   logarithmic score scaling.
4. **Results:** state the training length, evaluation range, and number of seeds
   in one metadata line, then use a large central figure and a smaller
   learned-log threshold plot to show how the modes diverge beyond training.
5. **Key takeaways:** state the two central conclusions and one scope note.
6. **References and acknowledgments:** place a short reference list and
   acknowledgment in the lower-right corner.

## Overall Layout

Use a conventional academic-poster structure without reproducing every report
section. The four-column grid should be combined into unequal visual blocks so
that the main result occupies the physical center of the poster.

```text
+------------------+--------------------------------+------------------+
| TITLE, SUBTITLE, AUTHORS, AND AFFILIATION                             |
+------------------+--------------------------------+------------------+
|                  | MODEL AND SCORE SCALING        | LEARNED-LOG      |
| MOTIVATION AND   |                                | THRESHOLD        |
| RESEARCH         | Constant | Log | Learned log   |                  |
| QUESTION         |                                | c Delta versus   |
|                  |                                | epochs           |
| WHY FIXED        +--------------------------------+------------------+
| ATTENTION        | RESULTS                        | KEY TAKEAWAYS    |
| DILUTES          | Train n=10 | Test to 10^7      |                  |
|                  |                                | Two conclusions  |
| Task visual      | Target attention versus       | One scope note   |
|                  | sequence length               +------------------+
| Closed-form      |                                | REFERENCES       |
| equation         | Large central figure           | ACKNOWLEDGMENTS  |
|                  |                                |                  |
+------------------+--------------------------------+------------------+
```

Layout allocation:

- The left column contains Motivation and Research Question above the Section
  1 task visual and closed-form explanation.
- The middle two columns contain Section 2 in the upper row, and the Results
  metadata line and large main result in the lower row.
- The right column contains the learned-log threshold plot, Key Takeaways, the
  scope statement, References, and Acknowledgments.
- Do not use a full-width References or Acknowledgments footer.
- Keep the main result centered across the middle two columns rather than
  placing all figures at the far right.

Sections intentionally omitted or combined:

- Do not include a separate Abstract; the header and Motivation block perform
  that role.
- Combine Introduction and Research Question into one short opening block.
- Use Section 1 as the only necessary Background rather than adding a general
  attention tutorial.
- Do not create a separate Experimental Design section; retain only the facts
  needed to interpret the figures in the Results metadata line and captions.
- Combine Conclusion and Takeaways into one block.
- Include the scope limitation as one sentence rather than a separate section.
- Do not create separate Related Work or Use of Artificial Intelligence
  sections unless an external requirement makes them necessary.

## Results Metadata Line

Place this compact line directly below the Results heading and above the main
figure in the middle two columns:

```text
TRAIN: n = 10 | EVALUATE: n = 10 to 10^7 | 5 RANDOM SEEDS
```

Render $n=10$ and $n=10$--$10^7$ with mathematical typesetting. Use
typographic separators or small icons, but do not give the line a separate
section box or heading. State the reporting convention in the main figure
caption:

> Means over five seeds; bands show $\pm1$ s.d.

The task definition and model schematic already establish the balanced binary
target-detection setting. Epoch budgets appear on the learned-log threshold
plot. Do not add the optimizer, learning rate, batch size, number of training
examples, initialization, or chunked-evaluation details to the poster.

## Lower-Right References and Acknowledgments

### References

> **[1]** Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.  
> **[2]** Press et al. (2022). *Train Short, Test Long: Attention with Linear
> Biases Enables Input Length Extrapolation.* ICLR.

Citation placement:

- Place **[2]** after the first Motivation sentence about performance beyond
  the training length.
- Place **[1]** after the reference to standard softmax attention in Section 1.

### Acknowledgments

> I thank Professor MacCormick for proposing the reduced-model direction and
> for guidance on the scope and presentation of this work.

### Visual Specification

- Place References and Acknowledgments below the Scope statement in the
  lower-right corner.
- Keep the two references readable rather than compressing them into footnote
  text.
- Do not include a QR code or a separate Related Work block.

### Status

- References: confirmed
- Acknowledgments: confirmed

## Motivation and Research Question

### Purpose

Establish why short-to-long generalization matters and lead directly into the
growing-competition mechanism without introducing technical notation.

### Poster Copy

#### Motivation

> Training on short sequences does not guarantee reliable performance on much
> longer inputs [2]. As sequence length grows, the target must compete with more
> non-target tokens for the same attention budget.

#### Research Question

> **How must attention sharpen with length to remain concentrated on the
> target?**

### Visual Specification

- Keep the motivation as two short sentences rather than a paragraph.
- Give the research question a distinct background or border so that it is the
  first text read within the left column.
- Do not introduce score-scaling notation here; Section 2 will connect
  attention sharpening to $\alpha(n)$.
- Place this block immediately above Section 1 so that the phrase "same
  attention budget" leads into the fixed-attention dilution visual.

### Status

- Copy: confirmed
- Visual treatment: not started

## Section 1: Why Fixed Attention Dilutes with Length

### Purpose

Introduce the target-detection task and establish why a fixed target advantage
is insufficient as the number of non-target competitors grows.

### Intended Takeaway

A higher target score than every individual non-target score is not sufficient
to preserve the target's share of attention as sequence length increases.

### Poster Copy

#### Heading

> **Why Fixed Attention Dilutes with Length**

#### Task Definition

> The model detects whether a designated target token is present in the
> sequence.

#### Main Explanation

> The target retains an advantage over every individual non-target, but the
> non-targets grow in number and dominate in aggregate.

#### Visual Labels

- Short sequence
- Long sequence
- Same target advantage
- Increasing aggregate non-target weight

### Formal Statement

Let $p_t(n)$ denote the share of attention assigned to the target token in a
length-$n$ sequence. With one target score $a$ and $n-1$ identical non-target
scores $b$, standard softmax attention [1] reduces to

```math
p_t(n)
=
\frac{\exp(a)}{\exp(a)+(n-1)\exp(b)}
=
\frac{1}{1+(n-1)\exp(-\Delta)},
\qquad
\Delta=a-b>0.
```

For any fixed target--non-target score margin $\Delta$, $p_t(n)\to0$ as the
number of non-target competitors grows without bound.

Equation takeaway:

> A fixed score margin separates the target from each non-target individually,
> but cannot offset their growing aggregate weight.

This expression should be presented as the closed form obtained by applying
standard softmax attention to the reduced two-score setting, not as a modified
attention rule.

### Visual Specification

- Use a horizontal short-sequence versus long-sequence comparison.
- Represent the target with one accent color and the non-targets in neutral
  gray.
- Keep the target's individual advantage unchanged in both scenes; only the
  number and aggregate weight of the non-targets should increase.
- Visually distinguish an individual non-target from the combined non-target
  weight, for example with a bracket or a grouped background shape.
- Link the visual to the equation by using the non-target color for the
  $(n-1)$ term and the target accent color for the score-margin annotation.
- Place the formal statement in a visually separate box so that the intuitive
  reading path does not require the equation.
- If illustrative numerical weights are used, label them as schematic rather
  than experimental measurements.
- Do not include query/key notation, model architecture details, scaling modes,
  or experimental curves in this section.

### Transition to Section 2

Possible bridge:

> What if the effective score margin grows with sequence length?

### Status

- Copy: first complete draft
- Formal statement: complete
- Visual: not started

## Reduced Model Schematic

### Purpose

Show how the reduced classifier converts a token sequence into a binary
decision, while exposing the two-score structure and the exact location of the
length-dependent multiplier. The schematic should establish the experimental
model without requiring query/key projection notation.

### Intended Takeaway

One final-position query scores the target and identical non-target tokens,
length-scaled softmax converts those scores into attention, and a linear
classifier reads the resulting attention-weighted sum.

### Schematic Flow

```text
TOKEN SEQUENCE
non-target  non-target  target  non-target  ...  non-target
      |
      v
FINAL-POSITION QUERY SCORES ALL TOKENS
S_n = (a, b, b, ..., b)
      |
      v
LENGTH-SCALED SOFTMAX
softmax(alpha(n) S_n)
      |
      v
ATTENTION-WEIGHTED SUM
o(n) = (p_t(n), 1 - p_t(n))
      |
      v
LINEAR CLASSIFIER
target absent / target present
```

Render the schematic horizontally in the poster. Use mathematical typesetting
for the following labels:

```math
S_n=(a,b,b,\ldots,b)
```

```math
\operatorname{softmax}\!\big(\alpha(n)S_n\big)
```

```math
o(n)=\big(p_t(n),\,1-p_t(n)\big)
```

### Poster Labels

1. **Token sequence**
2. **Final query scores all tokens**
3. **Length-scaled softmax**
4. **Attention-weighted sum**
5. **Linear classifier**

The final output labels are **Target absent** and **Target present**.

### Technical Note

> A single final-position query attends over one-hot token embeddings; the
> embeddings are reused as values, with no learned value projection.

The phrase **attention-weighted sum** should be used instead of **pooled
output**. Although the model reduces a variable-length sequence to a fixed-size
vector, it has no separate mean, max, or learned pooling layer.

### Visual Specification

- Place the schematic horizontally above the three Section 2 scaling cards in
  the middle two columns.
- Show one positive sequence with one accent-colored target token and neutral
  gray non-target tokens.
- Use the target score $a$ once and the repeated non-target score $b$ to make
  the two-score structure visible without displaying $W_Q$, $W_K$,
  $q_{\mathrm{last}}$, $k_t$, or $k_u$.
- Highlight the $\alpha(n)$ multiplier using the same accent treatment as the
  Section 2 intervention diagram.
- Keep arrows and labels visually dominant; place the technical note below the
  flow so that a general reader can skip it.
- Do not add a separate pooling operation, value projection, multi-head block,
  residual connection, or other full-transformer components.
- A negative-example path is not necessary; the task definition already states
  that the classifier predicts whether the target is present.

### Status

- Content: first complete draft
- Terminology: attention-weighted sum confirmed
- Visual: not started

## Section 2: How Length-Aware Scaling Counters Dilution

### Purpose

Introduce the score multiplier as the intervention and show how the three
scaling modes imply different long-length regimes. This section provides the
conceptual and mathematical bridge from the dilution mechanism in Section 1 to
the experimental evidence in Section 3.

### Intended Takeaway

Length-aware scaling preserves the score ordering while increasing the
effective separation seen by softmax. The long-length outcome depends on how
quickly this effective margin grows relative to the non-target competition.

### Poster Copy

#### Heading

> **How Length-Aware Scaling Counters Dilution**

#### Main Explanation

> Before softmax, all attention scores are multiplied by a positive factor
> $\alpha(n)$. The ranking is unchanged, but the effective score margin grows
> from $\Delta$ to $\alpha(n)\Delta$.

#### Intervention Diagram

```text
Score margin          Length-dependent scaling          Effective margin
    Delta         --------- x alpha(n) --------->        alpha(n) Delta
```

The rendered diagram should use mathematical notation for $\Delta$,
$\alpha(n)$, and $\alpha(n)\Delta$ rather than the plain-text placeholders
above.

### Scaling Modes

#### Constant

```math
\alpha(n)=1
```

> No response to sequence length

```math
p_t(n)\to0
```

#### Log

```math
\alpha(n)=\log n
```

> Preset logarithmic growth

```math
p_t(n)\to1
\qquad\text{if}\qquad
\Delta>1
```

#### Learned Log

```math
\alpha(n)=1+c\log(1+n)
```

> Optimization learns the positive growth coefficient $c$

```math
p_t(n)\to1
\qquad\text{if}\qquad
c\Delta>1
```

### Formal Criterion

Place the unifying condition in a visually separate box labeled
**Asymptotic criterion**:

```math
p_t(n)\to1
\quad\Longleftrightarrow\quad
\alpha(n)\Delta-\log(n-1)\to+\infty.
```

> The effective score margin must outgrow the logarithm of the number of
> non-target competitors.

Optional technical note:

> At equality, target attention approaches an intermediate value rather than
> one.

### Visual Specification

- Begin with a compact left-to-right transformation from $\Delta$ to
  $\alpha(n)\Delta$.
- State that the multiplier changes attention sharpness without changing the
  score ordering.
- Present Constant, Log, and Learned log as three horizontally aligned cards
  across the middle two columns.
- Give each mode one formula, one plain-language description, and one
  long-length outcome.
- Use the same mode colors that will appear in the Section 3 result figure.
- Keep the asymptotic criterion visually optional: it should be easy for a
  general reader to skip but large enough for a technical reader to inspect.
- Do not include optimization initialization, measured parameter values,
  classifier logits, failure-length calculations, or experimental accuracy in
  this section.
- Keep training and evaluation metadata out of the scaling cards; it belongs
  directly above the Section 3 result figure.

### Transition to Section 3

> **Same training-length accuracy. Sharply different long-length behavior.**

### Status

- Copy: first complete draft
- Formal criterion: complete
- Visual: not started

## Section 3: Results

### Purpose

Use the experimental measurements to verify the distinct long-length regimes
predicted in Sections 1 and 2. The main figure should show how target attention
diverges beyond the training length, while the smaller threshold figure should
show why finite benchmark accuracy does not identify the asymptotic regime.

### Intended Takeaways

1. Runs that fit the same short training length can develop sharply different
   target attention as sequence length increases.
2. The learned-log checkpoints can all pass the finite benchmark even though
   only the later checkpoints cross the theoretical growth threshold.

### Section Heading

> **Same Training-Length Accuracy. Sharply Different Long-Length Behavior.**

Place the Results metadata line immediately below this heading:

```text
TRAIN: n = 10 | EVALUATE: n = 10 to 10^7 | 5 RANDOM SEEDS
```

### Figure A: Target Attention Across Length

#### Figure Title

> **Target Attention Diverges Beyond the Training Length**

#### Representative Runs

- Constant (50)
- Log (50)
- Learned log (50), with $c\Delta=0.58<1$
- Learned log (200), with $c\Delta=1.14>1$

The four runs represent fixed-scaling collapse, predetermined logarithmic
scaling, learned logarithmic scaling below the asymptotic threshold, and
learned logarithmic scaling above the threshold. Omit the remaining budgets
because they repeat these regimes without adding a new comparison.

#### Axes and Marks

- Horizontal axis: sequence length $n$ on a logarithmic scale from $10$ to
  $10^7$
- Vertical axis: target attention $p_t(n)$ from 0 to 1
- Plot the mean curve and a $\pm1$ s.d. band over five seeds.
- Use a compact one-row legend above the plot. Direct labels are omitted because
  the overlapping successful curves and their connectors reduce the usable plot
  width.
- Distinguish curves with marker shape and line style as well as color.
- Do not extend the curves beyond the evaluated range; theoretical predictions
  belong in the threshold figure and callouts.

Recommended visual identities:

- Constant: charcoal, solid line, circle marker
- Log: blue, narrow dash-dot line, diamond marker
- Learned log below threshold: muted amber, dashed line, downward-triangle marker
- Learned log above threshold: reddish purple, wide solid line,
  upward-triangle marker

Include the learned-log epoch budgets and $c\Delta$ values in the legend. Since
Log (50) and Learned log (200) overlap near $p_t(n)=1$, so draw the latter as
a wider solid line beneath the narrower dash-dot Log line and alternate their
marker positions. This keeps both series visible along the shared path.

#### Figure A Callout

Callout heading:

> **Same Margin, Different Outcome**

Callout copy:

> Constant (50) and Learned log (200) both learn $\Delta\approx9$, yet their
> positive accuracy at $n=10^7$ is 0% versus 100%.

This comparison isolates length scaling from the raw learned score margin.

#### Figure A Caption

> Representative runs evaluated from the training length to $10^7$. Curves
> show means over five seeds; bands show $\pm1$ s.d.

### Figure B: Learned-Log Threshold Crossing

#### Figure Title

> **Learned Scaling Crosses the Theoretical Threshold**

#### Axes and Marks

- Horizontal axis: training checkpoints at 50, 100, 200, and 400 epochs
- Vertical axis: learned growth rate $c\Delta$
- Plot means with $\pm1$ s.d. error bars over five seeds.
- Label the four mean values: 0.58, 0.83, 1.14, and 1.55.
- Draw a prominent horizontal threshold at $c\Delta=1$.
- Connect the ordered checkpoints with a thin line, but refer to them as
  checkpoints rather than converged solutions.

#### Figure B Callout

Callout heading:

> **A Finite Pass Can Hide Later Failure**

Callout copy:

> Every learned-log checkpoint reaches 100% accuracy at $n=10^7$, but only the
> 200- and 400-epoch checkpoints satisfy $c\Delta>1$.

Optional secondary line:

> The earlier checkpoints are predicted to fail at still greater lengths.

#### Figure B Caption

> The learned growth rate crosses $c\Delta=1$ between the 100- and 200-epoch
> checkpoints. Points show means over five seeds; error bars show $\pm1$ s.d.

### Placement

- Place Figure A across the middle two columns as the largest visual on the
  poster.
- Place Figure B in the upper portion of the right column.
- Keep the Figure A callout adjacent to or immediately below the main plot.
- Place the Figure B callout directly below the threshold plot so that it leads
  into the Key Takeaways block.

### Excluded Result Details

- Positive-example logit panel
- Full results table
- All eight runs in one legend
- Per-seed values
- Classifier threshold $p^{\ast}$
- Predicted failure-length equation
- Extrapolated curves beyond $10^7$

These details remain available in the full report and should not compete with
the two poster claims.

### Status

- Figure selection: confirmed
- Callout copy: first complete draft
- Figure generation: first draft generated

Generated assets:

- `figures/poster_target_attention_by_length.pdf` and PNG preview
- `figures/poster_learned_log_threshold.pdf` and PNG preview
- Generation script: `../../src/make_poster_figures.py`

## Key Takeaways and Scope

### Purpose

Answer the research question with two concise conclusions, then state the
boundary of the claim without repeating the finite-benchmark callout from the
Results section.

### Poster Copy

#### Takeaway 1

> **Fixed scaling postpones, but cannot prevent, target-attention dilution.**
> Any fixed target--non-target score margin is eventually overwhelmed by the
> growing aggregate non-target weight.

#### Takeaway 2

> **Sufficiently strong logarithmic sharpening can prevent dilution.** Target
> attention converges to one when $\Delta>1$ for Log or $c\Delta>1$ for Learned
> log.

#### Scope

> **These thresholds are exact for the reduced two-score classifier studied
> here; whether analogous conditions hold in full transformers remains open.**

### Visual Specification

- Place this block below the Figure B callout in the right column.
- Use two numbered takeaways rather than a paragraph or a separate Conclusion
  section.
- Set the bold lead sentence of each takeaway larger than its supporting
  sentence.
- Keep the threshold conditions on the supporting line so that a general reader
  can understand the claim without reading the notation.
- Separate the Scope statement with a light rule or subtle background, but do
  not reduce it to footnote-sized text.
- Do not repeat **A Finite Pass Can Hide Later Failure** here; the Results
  callout already delivers that supporting message.
- Do not add a limitations list or future-work bullets.

### Status

- Takeaway copy: confirmed
- Scope copy: confirmed
- Visual: not started
