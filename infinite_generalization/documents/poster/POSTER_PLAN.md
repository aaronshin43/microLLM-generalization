# Poster Plan

## Format and Audience

- Size: 40 x 30 inches
- Orientation: landscape
- Grid: four-column-inspired layout with three unequal reading regions
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

> Same training-length accuracy. Different long-length behavior.

## Header

### Main Title

> **How Score Scaling Shapes Attention Beyond the Training Length**

### Author and Affiliation

> **Aaron Shin '27**
>
> Deep Network Understanding Lab
>
> Department of Mathematics and Computer Science, Dickinson College

### Advisor

> Advisor: Prof. John MacCormick

### Visual Specification

- Set the main title as the largest text on the poster, reduced from 78 pt to
  64 pt so that it fits between the logos without wrapping.
- Omit the subtitle; the main title now identifies score scaling directly.
- Center all header lines. Place the author and advisor on one shared line below
  the title, separated by a subtle vertical divider.
- Keep the author bold and the advisor slightly smaller, explicitly labeled
  Advisor rather than implying coauthorship.
- Use 32 pt for the author, 28 pt for the advisor, and 26 pt for both affiliation
  lines, a 2 pt increase for each while preserving the title and logo sizes.
- Place the lab on the next line, followed by the department and college together
  on the final line, moving from the specific research group to the institution.
- Do not add another central-claim sentence to the header.
- Use three vertically centered header columns: Dickinson wordmark on the left,
  title and author/affiliation block in the middle, and DNU Lab logo on the right.
  Reserve 12%, 76%, and 12% of the header's inner width, respectively.
- Preserve both logos' original colors and proportions. Use the transparent
  wordmark directly on the navy background at 4.10 inches wide, with no white
  panel. Enlarge the square DNU logo from 1.50 to 2.35 inches high.
- Increase the outer header height from 3.25 to 3.80 inches and vertically center
  its contents in the enlarged card. Set the header-to-body gap explicitly to
  0.28 inches, absorbing the former excess gap into the header while preserving
  the body placement and layout.
- Use `assets/dickinson_wordmark.png` and `assets/dnu_logo.pdf`, the latter a
  vector export of the supplied `assets/dnu_logo.svg`.

### Status

- Main title: confirmed
- Subtitle: removed
- Author and affiliation: confirmed
- Advisor: explicit role placed beside the author
- Logos: transparent Dickinson wordmark and enlarged DNU logo flank the full
  title and affiliation block

## Overall Storyboard

1. **Motivation and research question:** state the short-to-long generalization
   problem and ask when length-aware attention prevents target dilution.
2. **Task & Setup and Why Attention Dilutes:** introduce the two-token
   detection task and reduced model in one card, then use a separate card to
   contrast an individual target advantage with growing non-target competition.
3. **Model and score scaling:** introduce constant, logarithmic, and learned
   logarithmic score scaling.
4. **Results:** state the training length, evaluation range, and number of seeds
   in one metadata line, then use a large central figure to show how the modes
   diverge beyond training.
5. **Attention in Closed Form:** collect the optional mathematical explanation
   in an upper-right card, using the scaled softmax expression for all modes.
6. **Conclusion and Scope:** state the two central conclusions, contrast an
   observed finite pass with predicted eventual failure, and give the scope
   of the analysis in a separate card.
7. **References and acknowledgments:** place a short reference list and
   acknowledgment in the lower-right corner.

## Overall Layout

Use a conventional academic-poster structure without reproducing every report
section. The four-column grid should be combined into unequal visual blocks so
that the main result occupies the physical center of the poster.

```text
+------------------+--------------------------------+------------------+
| TITLE, AUTHOR, ADVISOR, AND AFFILIATIONS                              |
+------------------+--------------------------------+------------------+
| MOTIVATION AND   | MODEL AND SCORE SCALING         | ATTENTION IN     |
| RESEARCH QUESTION|                                | CLOSED FORM      |
|                  | Constant | Log | Learned log   | Scaled softmax   |
|                  |                                +------------------+
| TASK & SETUP     |                                | CONCLUSION       |
| Task + model     +--------------------------------+ Two conclusions  |
|                  | RESULTS                        |                  |
|------------------| Train n=10 | Test to 10^7       | Observed pass /  |
| WHY ATTENTION    |                                | predicted failure|
| DILUTES          | Target attention versus        +------------------+
|                  | sequence length                | SCOPE            |
| Short/long visual|                                |                  |
|                  | Large central figure           +------------------+
|                  |                                | REFERENCES &     |
|                  |                                | ACKNOWLEDGMENTS  |
+------------------+--------------------------------+------------------+
```

Layout allocation:

- The left column contains three cards: Motivation with its integrated
  research question, Task & Setup with the input examples and model sentence,
  and Why Attention Dilutes with the short/long schematic and its intuitive
  explanation. The model sentence is ordinary body text below the examples,
  without a Model subheading.
- The middle two columns contain Section 2 in the upper row, and the Results
  metadata line and large main result in the lower row.
- The right column contains Attention in Closed Form, Conclusion, Scope, and
  References & Acknowledgments as four separate cards. The learned-log
  threshold plot is omitted.
- Do not use a full-width References or Acknowledgments footer.
- Keep the main result centered across the middle two columns rather than
  placing all figures at the far right.

Implementation status:

- Refined 40 x 30 inch full-layout draft assembled in `latex/POSTER.tex`.
- The four-column grid is implemented as three reading regions: 9.30 inches
  on the left, 20.40 inches in the center, and 8.30 inches on the right, with
  0.45-inch gutters. The right region was narrowed by one inch to give the
  central results more space after removing the threshold plot.
- The three reading regions retain segmented section cards. Card heights vary
  with content, but their totals and lower baselines are matched across the
  three regions.
- Second-pass text reduction: Motivation and the research question share one
  opening card, repeated claim statements collapse into single punchline
  boxes, in-figure titles are removed, captions state only the seed and band
  convention, and the schematic technical note and the Section 1 bridge line
  are dropped.
- Second-pass visual consistency: punchline statements share one navy
  left-bar insight style, structural accents (heading rules, schematic arrows,
  highlighted softmax stage) use navy, the target token is navy in every
  diagram, and the references card is sized to its content.
- Third pass: card interiors use fixed-height regions with flexible gaps so
  leftover space spreads between blocks instead of pooling at card bottoms,
  and the Section 1 visual gained a per-token weight-bar layer that shows the
  unchanged individual advantage alongside the shrinking share.
- Left-column organization: use three outer cards, with Motivation at 4.15
  inches, Task & Setup at 5.75 inches, and Why Attention Dilutes at 14.34
  inches, separated by 0.28-inch gaps. Preserve the original column height
  and bottom edge. Card headings use 34 pt type and body copy remains 26 pt.
  Neither Task nor Model has a separate subheading. Distribute the extra
  0.60 inches in Task & Setup around the input examples and model sentence;
  reduce Why Attention Dilutes by the same amount without shrinking its graphic.
- Keep the fixed-score-advantage explanation in regular 24 pt type with 29 pt
  leading above the schematic, with no box or left rule. Move the softmax
  formula out of this card. Increase the gap between the short/long scenes
  by 1.28 inches before scaling, enlarge the whole schematic uniformly by
  6%, and increase the base scene-label, note, and share-bar-label sizes to
  25 pt, 20 pt, and 17 pt, respectively. Its illustrative bar proportions
  are preserved by the uniform enlargement.
- Right-column revision: use Attention in Closed Form (4.80 inches),
  Conclusion (9.20 inches), Scope (4.70 inches), and References &
  Acknowledgments (5.26 inches), separated by 0.28-inch gaps. The formula
  card uses the same white background and neutral border as the other section
  cards, with a 36 pt equation. The two main
  conclusions use 34 pt type with selective bold emphasis. A plain 26 pt
  paragraph contrasts observed accuracy with predicted eventual failure for
  Learned log (50), without a separate heading or comparison boxes. Scope
  uses ordinary 26 pt body text; references and
  acknowledgments use 23 pt with 28 pt leading. Redistribute 1.20 inches
  from Conclusion to Scope (0.50 inches) and References & Acknowledgments
  (0.70 inches). All three regions retain the same lower edge.
- The main figure is now 19.20 inches wide. Its data, annotations, and vector
  asset are unchanged; only its displayed size increases with the wider
  center region.

Sections intentionally omitted or combined:

- Do not include a separate Abstract; the header and Motivation block perform
  that role.
- Combine Introduction and Research Question into one short opening block.
- Use Section 1 as the only necessary Background rather than adding a general
  attention tutorial.
- Do not create a separate Experimental Design section; retain only the facts
  needed to interpret the figures in the Results metadata line and captions.
- Present the takeaways under Conclusion, without a separate Key Takeaways card.
- Give Scope its own compact card with two sentences rather than a list of
  detailed limitations.
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
target-detection setting. Run labels retain the epoch budgets in parentheses.
Do not add the optimizer, learning rate, batch size, number of training
examples, initialization, or chunked-evaluation details to the poster.

## Lower-Right References and Acknowledgments

### References

> **[1]** Press et al. (2022). *Train Short, Test Long: Attention with Linear
> Biases Enables Input Length Extrapolation.* ICLR.
>
> **[2]** Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.

Citation placement:

- Number references in order of first appearance in the poster's reading flow.
- Place **[1]** after the first Motivation sentence about performance beyond
  the training length.
- Place **[2]** after standard softmax in the upper-right Attention in Closed
  Form card.

### Acknowledgments

> I thank Professor MacCormick for proposing the reduced-model direction and
> for guidance on the scope and presentation of this work.

### Visual Specification

- Place References and Acknowledgments below the Scope statement in the
  lower-right corner.
- Use a 5.26-inch-high card with 23 pt text and 28 pt leading.
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
> longer inputs [1]. As sequence length grows, more non-target tokens compete
> for the same attention budget. We ask how attention must sharpen with length
> to remain concentrated on the target.

### Visual Specification

- Keep all three sentences in one continuous paragraph, with no forced line
  break before the research question and no separate question box or heading.
- Match the final sentence to the ordinary body text in color, weight, and
  size; do not emphasize it separately.
- Do not introduce score-scaling notation here; Section 2 will connect
  attention sharpening to $\alpha(n)$.
- Place this block immediately above Section 1 so that the phrase "same
  attention budget" leads into the task setup and attention-dilution visual.

### Status

- Copy: confirmed
- Visual treatment: the research question continues the same paragraph in
  ordinary body styling, with no separate box or Research Question heading

## Section 1: Task & Setup and Why Attention Dilutes

### Purpose

Introduce the target-detection task and identify the reduced model, then
establish why a fixed target advantage is insufficient as the number of
non-target competitors grows. Put the task and model in a Task & Setup card,
followed by a separate Why Attention Dilutes card for the visual and intuitive
explanation. The equation is in the upper-right Attention in Closed Form card.

### Intended Takeaway

A higher target score than every individual non-target score is not sufficient
to preserve the target's share of attention as sequence length increases.

### Poster Copy

#### Heading

> **Task & Setup**

#### Task

Begin directly with the task statement; do not repeat Task as a visible
subheading beneath the card title.

> Detect whether a target token (**T**) is present. All other tokens are
> identical non-targets (**N**).

Render the two examples as compact token diagrams using navy for T and gray
for N, with an arrow to the binary label:

```text
T N N ... N  ->  Target present
N N N ... N  ->  Target absent
```

#### Model

This label organizes the plan only; render the model sentence as an ordinary
paragraph below the examples, without a visible subheading.

> We use a reduced attention-based binary classifier with one
> readout query, fixed one-hot token values, and a linear output layer.

This sentence identifies the model without duplicating the central Method
schematic. Do not add a parameter-count table or describe the architecture as
a one-layer Transformer.

#### Why Attention Dilutes

Use this as a standalone card heading, followed by the fixed-score-advantage
explanation in ordinary type, then the existing short/long attention schematic.

The former prose sentence ("The target retains an advantage over every
individual non-target, but the non-targets grow in number and dominate in
aggregate") is delivered visually. Each scene stacks three layers: a per-token
attention bar chart, the token row, and a target-share bar. In the long scene,
every normalized per-token bar is shorter because the fixed attention budget is
split across more tokens, while the target remains taller than any individual
non-target. The growing collection of small non-target bars and the shrinking
target segment in the share bar show the aggregate loss. The long token row
uses an ellipsis but keeps the final non-target visible because the query comes
from that last position. The short scene uses the actual training length
$n=10$ and an illustrative target share of 75%; the long scene uses the
same fixed-score setting and a target share of about 22%. The only
remaining prose claim is the short explanation above the visual. All per-token bars use one
qualitative display scale rather than encoding exact numeric proportions. The
non-target bars are deliberately enlarged and use a darker neutral fill so that
their small weights remain visible in print.

#### Visual Labels

- Short sequence ($n=10$) / Long sequence
- Attention weight per token
- many small weights add up (brace over the long scene's non-target bars)
- target's share of attention / non-targets (short share bar)
- target / all non-targets combined (long share bar)
- (schematic)

Keep the short share bar's total width unchanged, allocating 75% to the target
and 25% to non-targets. Center each label within its segment. Do not print
these illustrative percentages or change the per-token bars or long scene.

### Introductory Explanation

> A fixed score advantage over each non-target cannot prevent the target's
> share of attention from shrinking as the sequence grows.

The bars represent normalized attention weights, not raw scores or their
exponentials. The left card retains only the plain-language phrase "score
advantage." Define $a$, $b$, and $\Delta=a-b$ in the central scaling
explanation; show the equation in the upper-right card.

### Visual Specification

- Stack the short-sequence and long-sequence scenes vertically below the
  Why Attention Dilutes card heading and introductory explanation.
- Represent the target with one accent color and the non-targets in neutral
  gray.
- Keep the target taller than each individual non-target in both scenes,
  while shrinking all individual normalized weights in the long scene.
  Only the number and combined attention of the non-targets should increase.
- Visually distinguish an individual non-target from the combined non-target
  weight, for example with a bracket or a grouped background shape.
- Keep score notation in the formal statement rather than labeling the
  normalized attention bars with $a$ and $b$.
- Set the fixed-score-advantage explanation in regular 24 pt type above the
  schematic, without a box or left rule. Keep this card equation-free so that
  the intuitive reading path does not require mathematical notation.
- Use the freed space to separate the short/long scenes and slightly enlarge
  the schematic and its labels, without changing the illustrative proportions.
- If illustrative numerical weights are used, label them as schematic rather
  than experimental measurements.
- Limit the model introduction to one ordinary paragraph below the task
  examples, without a separate Model subheading. Leave the
  query/key notation, detailed model pathway, scaling modes, and experimental
  curves to the central Method and Results cards.

### Transition to Section 2

No bridge line is used. The research question in the opening card already asks
it, and the Section 2 heading answers it.

### Status

- Copy: task definition, two compact labeled examples, one model sentence,
  and the intuitive explanation above the dilution schematic
- Card structure: Task & Setup and Why Attention Dilutes are separate cards;
  the schematic is slightly enlarged, and the introductory explanation uses
  regular type without a box
- Formal statement: moved to the upper-right card and generalized to include
  $\alpha(n)$; score notation is defined in the central Method card
- Visual: refined draft; each scene stacks normalized per-token attention bars,
  the token row, and a share bar, with an ellipsis in the long sequence and a
  navy target throughout

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
2. **Last-position query scores all tokens**
3. **Length-scaled softmax**
4. **Attention-weighted sum** *(one-hot values)*
5. **Linear classifier**

The final output labels are **Target absent** and **Target present**.

### Technical Note

> A single final-position query attends over one-hot token embeddings; the
> embeddings are reused as values, with no learned value projection.

The schematic now identifies the last-position query and the one-hot values
directly. The remaining implementation detail---that there is no learned value
projection---is omitted on the poster and remains in the full report.

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
- Highlight the length-scaled softmax stage with a heavier navy outline; the
  technical note is omitted on the poster.
- Keep arrows and labels visually dominant; arrows use the structural navy.
- Enlarge the schematic slightly within the center card and increase the
  in-node type so that the five-stage pathway remains legible at poster-viewing
  distance.
- Do not add a separate pooling operation, value projection, multi-head block,
  residual connection, or other full-transformer components.
- A negative-example path is not necessary; the task definition already states
  that the classifier predicts whether the target is present.

### Status

- Content: first complete draft; the last-position query and one-hot values are
  labeled directly, while the no-value-projection detail is omitted
- Terminology: attention-weighted sum confirmed
- Visual: refined draft with enlarged nodes and type, navy arrows, and a
  navy-outlined softmax stage

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

> Learned query and key vectors produce the pre-softmax scores $a$ (target) and
> $b$ (non-target) through scaled dot products. Their difference is the score
> margin $\Delta=a-b$. Multiplying every score by a positive factor $\alpha(n)$
> preserves the score ranking and changes this margin to $\alpha(n)\Delta$.

### Scaling Modes

Shared note above the three mode cards:

> Attention limits as sequence length $n\to\infty$

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

- State that the multiplier changes attention sharpness without changing the
  score ordering.
- Present Constant, Log, and Learned log as three horizontally aligned cards
  across the middle two columns.
- Give each mode one formula, one plain-language description, and one
  long-length outcome.
- Add a single centered 21 pt note above the three cards to identify the
  displayed limits as sequence-length limits, rather than training limits.
- Fix each mode title to the top of its card, then vertically center the
  formula--description--outcome group in a separate fixed-height region below
  it.
- Use the same mode colors that will appear in the Section 3 result figure.
- Keep the asymptotic criterion visually optional: it should be easy for a
  general reader to skip but large enough for a technical reader to inspect.
- Do not include optimization initialization, measured parameter values,
  classifier logits, failure-length calculations, or experimental accuracy in
  this section.
- Keep training and evaluation metadata out of the scaling cards; it belongs
  directly above the Section 3 result figure.

### Transition to Section 3

> **Same training-length accuracy. Different long-length behavior.**

### Status

- Copy: first complete draft
- Formal criterion: complete
- Visual: refined full-layout draft complete; mode titles share a fixed top
  baseline and each explanatory group is vertically centered below its title

## Section 3: Results

### Purpose

Use the experimental measurements to verify the distinct long-length regimes
predicted in Sections 1 and 2. The main figure shows how target attention
diverges beyond the training length. The final Conclusion paragraph explains why
finite benchmark accuracy does not establish long-length behavior.

### Intended Takeaways

1. Runs that fit the same short training length can develop sharply different
   target attention as sequence length increases.
2. Learned log (50) passes the finite benchmark even though its learned
   parameters imply eventual failure; Learned log (200) lies above the
   theoretical growth threshold.

### Section Heading

> **Same Training-Length Accuracy, Different Long-Length Behavior**

Place the Results metadata line immediately below this heading:

```text
TRAIN: n = 10 | EVALUATE: n = 10 to 10^7 | 5 RANDOM SEEDS
```

### Figure A: Target Attention Across Length

#### Figure Title

The figure carries no internal title; the Results card heading delivers the
message and the former title is not repeated inside the plot.

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
  belong in the final Conclusion paragraph, explicitly identified as theoretical.

Recommended visual identities:

- Constant: charcoal, solid line, circle marker
- Log: blue, narrow dash-dot line, diamond marker
- Learned log below threshold: muted amber, dashed line, downward-triangle marker
- Learned log above threshold: reddish purple, wide solid line,
  upward-triangle marker

Include the learned-log epoch budgets and $c\Delta$ values in the legend. Since
Log (50) and Learned log (200) overlap near $p_t(n)=1$, draw the latter as
a wider solid line beneath the narrower dash-dot Log line and alternate their
marker positions. This keeps both series visible along the shared path.

#### Figure A Callout

The callout is rendered inside the plot's empty central region as a navy
annotation with two thin arrows to the diverging curve groups:

> **Same margin, opposite outcomes**  
> Constant (50) vs Learned log (200)
> both learn $\Delta\approx9$  
> positive accuracy at $n=10^7$: 0% vs 100%

The former closing sentence ("This comparison isolates length scaling from the
raw learned score margin") is dropped.

#### Figure A Caption

> Means over five seeds; bands show $\pm1$ s.d.

The evaluated range is not repeated here; the Results metadata line already
states it.

### Omitted Learned-Log Threshold Plot

The threshold plot is no longer displayed on the poster. The main figure's
legend retains the learned growth rates for the two representative learned-log
runs. Its former finite-benchmark message moves into a plain paragraph in
Conclusion that distinguishes observation from prediction. Keep the existing threshold figure assets available;
removing the section does not require deleting or regenerating those files.

### Placement

- Place Figure A across the middle two columns as the largest visual on the
  poster.
- The Figure A callout lives inside the plot as an annotation.
- Put Attention in Closed Form at the upper right, followed by Conclusion,
  Scope, and References & Acknowledgments at the lower-right corner.

### Excluded Result Details

- Positive-example logit panel
- Full results table
- All eight runs in one legend
- Per-seed values
- Classifier threshold $p^{\ast}$
- Predicted failure-length equation
- Extrapolated curves beyond $10^7$
- The separate learned-log threshold plot across training budgets

These details remain available in the full report and should not compete with
the two poster claims.

### Status

- Figure selection: the main target-attention figure only
- Callout copy: the margin comparison remains inside the main figure; the
  finite-pass caution appears as a plain paragraph in Conclusion
- Figure assets: unchanged; the main figure is displayed at 19.20 inches wide
  and the threshold plot is no longer embedded

Generated assets:

- `figures/poster_target_attention_by_length.pdf` and PNG preview
- `figures/poster_learned_log_threshold.pdf` and PNG preview (retained, not used)
- Generation script: `../../src/make_poster_figures.py`

## Attention in Closed Form

### Purpose

Provide an optional mathematical summary after the reader encounters the
model, scaling modes, and main result. Use one equation that includes all
three modes, rather than placing the constant-scaling special case beside
the introductory dilution schematic.

### Poster Copy

> Standard softmax [2] gives the target's attention share in a positive
> sequence of length $n$:

```math
p_t(n)=\frac{e^{\alpha(n)a}}
{e^{\alpha(n)a}+(n-1)e^{\alpha(n)b}}.
```

The central Method paragraph defines $a$ as the target's pre-softmax score,
$b$ as each non-target's score, and $\Delta=a-b$ as their margin. A positive
sequence has one target and $n-1$ identical non-targets, so the denominator
adds the target's unnormalized weight and the aggregate non-target weight.
The equation is standard softmax applied to the scaled scores in this model,
not a different attention rule. It is valid without assuming $a>b$; the
constant-scaling illustration on the left depicts a fixed positive advantage.

### Visual Specification

- Place this 4.80-inch-high card at the top of the right column.
- Use the standard white background and neutral gray border shared by the
  other section cards, without a nested box.
- Set the equation at 36 pt, substantially larger than the former 21 pt
  left-column version. Use 24 pt for the explanatory sentence. Omit the
  constant-scaling note because the central mode card already defines it.
- Do not add an equivalent reciprocal or margin-based formula, repeat the
  asymptotic criterion, or repeat the mode-specific thresholds here.
- Keep the central Method card's scaling conditions and the main result
  figure unchanged.

### Status

- Formula: includes the length-dependent multiplier for all three modes
- Placement: upper-right card; removed from the left-column schematic
- Reading layers: intuitive explanation on the left, notation and scaling
  in the center, optional large equation on the right

## Conclusion

### Purpose

Answer the research question with two prominent conclusions and preserve the
finite-benchmark caution after removing the threshold plot. Do not repeat the
central scaling formulas or add new result claims.

### Poster Copy

#### Main Conclusion 1

> Even a large score advantage **cannot prevent attention dilution** under
> constant scaling.

#### Main Conclusion 2

> **Sufficiently strong logarithmic scaling** makes the target receive nearly
> all attention as sequences grow.

#### Supporting Paragraph

> Finite success can still hide later failure. Learned log (50) reaches 100%
> accuracy at ten million tokens, yet the theory predicts eventual failure.

The observed accuracy is a measured result. The eventual failure is a
closed-form prediction, not an experiment beyond the evaluated range. Do not
imply that the downward attention curve alone proves eventual classification
failure.

### Visual Specification

- Use one 9.20-inch-high Conclusion card below Attention in Closed Form.
- Set the two bulleted conclusions in 34 pt type with 41 pt leading and
  selective bold emphasis, not all-bold paragraphs.
- Keep the first conclusion close to the heading; distribute the remaining
  space between the two main conclusions and the supporting paragraph.
- Set the closing paragraph in ordinary 26 pt body text with 31 pt leading.
  Remove its separate heading, run label, colored boxes, and all-bold styling.
  Keep the observation and theoretical prediction explicit within the prose.
- Do not repeat $\Delta>1$ or $c\Delta>1$ here; both are already visible in the
  central Method card.

### Status

- Copy: confirmed
- Visual: two prominent conclusions followed by one ordinary closing paragraph
- Threshold plot: removed from the poster, with its central caution retained

## Scope

### Purpose

State where the exact conclusions apply without adding a detailed caveat list
or implying that the reduced model establishes a result for full transformers.

### Poster Copy

> The exact analysis applies to a reduced classifier with one target per positive
> sequence, identical non-targets, and fixed one-hot values.
>
> Whether the same scaling conditions extend to full transformers remains open.

### Visual Specification

- Place Scope in a separate 4.70-inch-high card below Conclusion.
- Use ordinary 26 pt body text. Separate the setting and transfer-boundary
  sentences with a 0.32-inch additional paragraph gap, leaving more bottom
  padding within the enlarged card.
- Keep optimizer, identifiability, and convergence caveats in the report.
- Retain References & Acknowledgments below this card; there is enough room
  to keep the acknowledgment.

### Status

- Copy: confirmed
- Visual: standalone Scope card, with References & Acknowledgments retained
