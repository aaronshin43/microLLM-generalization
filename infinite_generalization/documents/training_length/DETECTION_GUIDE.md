# Binary Detection Guide

This guide connects the two-token reduced binary classifier in the [final report](../FINAL_REPORT.md) to the current implementation. The next task is to compare training lengths while retaining detection. Placing the target first is a dataset convention, not the FIRST task of classifying an arbitrary first symbol. See [BASELINE_REPRODUCTION.md](BASELINE_REPRODUCTION.md) for artifact provenance and commands, and [PLAN.md](PLAN.md) for experiment candidates.

## File map and entry point

Paths are relative to `infinite_generalization/`.

| Role | File or location | Connection |
|---|---|---|
| Research definition | [FINAL_REPORT.md](../FINAL_REPORT.md) | Two-score model, seeds 0–4, report table/figures |
| Historical experiments | [Stage 3 / 3B](../STAGE3_SIMPLIFIED_LENGTH_AWARE_ATTENTION.md) | Single-seed baselines and existing multi-length exploration |
| Weight interpretation | [Weight-level mechanism](../STAGE3_WEIGHT_LEVEL_MECHANISM.md) | A separate seed 42 rerun's query/key decomposition |
| Execution | [stage3_simplified_attention.py](../../src/stage3_simplified_attention.py) | `main → build_config → train_model → evaluate_length` |
| Configuration | `Stage3Config`, `parse_args`, saved `config.json` | Stage 3 accepts CLI arguments, with no YAML `--config` |
| Report results | `runs/stage3_seeds/<condition>_s<seed>/` | Config, history, metrics and checkpoint |
| Analysis | [analyze_stage3_mechanism.py](../../src/analyze_stage3_mechanism.py) | Checkpoint-derived query/key margin decomposition |
| Report figure | [make_report_figures.py](../../src/make_report_figures.py) | Report-family CSVs → attention/logit figure |
| Evaluation tests | [test_stage3_eval_reliability.py](../../tests/test_stage3_eval_reliability.py) | Chunking, class counts and stratification |

After [README](../../README.md) environment setup, inspect `python -m stage3_simplified_attention --help`. An argument-free run starts 200 epochs and evaluation up to 10M; use the bounded commands in the reproduction record for a short check. Relative `--output-dir` paths resolve under the code's `project_dir()`, not the shell's working directory.

## Data and the final query

The report baseline uses `target_token_count=1`, `non_target_token_count=1`, `target_position_mode=fixed_start`, and `d_head=2`. Target id is 0 and non-target id is 1. At length 10:

```text
positive: 0 1 1 1 1 1 1 1 1 1    label = 1
negative: 1 1 1 1 1 1 1 1 1 1    label = 0
```

Positives contain exactly one target; negatives contain none. `make_two_token_dataset` creates `examples // 2` positives and the remaining negatives; an odd count has one extra negative. `make_stage3_dataset_from_counts` keeps the final token non-target and shuffles examples with a local generator. In the baseline, all sequences within each class are identical.

The final non-target query provides a consistent readout slot without revealing the label. Its token also participates as a key/value. There is no positional encoding, causal mask, residual, feedforward block or max pooling. `nonfinal_random` places targets anywhere except the final position. Permuting the first $n-1$ tokens leaves the output unchanged by construction.

## Forward: input to loss

For batch size $B$, sequence length $n$ and query/key width $d$, the baseline vocabulary has size 2. `SimplifiedLastQueryAttentionClassifier.forward` performs:

| Step | Implementation | Shape |
|---|---|---|
| Token ids | Loader's `tokens` | `[B, n]`, int64 |
| Keys from fixed one-hot inputs | `project_tokens(tokens, key_projection)` | `[B, n, d]` |
| Final query only | `project_tokens(tokens[:, -1], query_projection)` | `[B, d]` |
| Raw scores | `einsum("bd,bld->bl", ...) / sqrt(d)` | `[B, n]` |
| Scaling | `alpha_for_length(n) * raw_scores` | `[B, n]` |
| Softmax | Along the last dimension | `[B, n]` |
| Fixed value readout | `token_value_output` | `[B, 2]` |
| Binary head | `Linear(2, 1)` and squeeze | `[B]` |
| Loss | `BCEWithLogitsLoss` with float labels | Scalar batch mean |

No sequence one-hot tensor is materialized. `F.embedding` selects a projection weight column. PyTorch stores each projection as `[d, 2]`; the report's row-vector $XW_Q$ notation uses the transpose `[2,d]`. These are the same operation. The code computes only one query and never builds the full $n\times n$ attention matrix.

```math
q_u=W_Q^{\mathrm{PT}}[0,1]^\top,\quad
k_t=W_K^{\mathrm{PT}}[1,0]^\top,\quad
k_u=W_K^{\mathrm{PT}}[0,1]^\top,
\qquad s_j=\frac{q_u^\top k_j}{\sqrt d}.
```

Values are fixed: target `[1,0]`, non-target `[0,1]`. There is no learned embedding or value projection. The current binary extension sums attention over all target types into the first coordinate.

```math
A_j=\operatorname{softmax}(\alpha(n)s)_j,\qquad
o_{\mathrm{pos}}=[p_t,1-p_t],\qquad o_{\mathrm{neg}}=[0,1],
\qquad z=(w_t-w_u)p_t+(w_u+\beta).
```

Training passes logits directly into the loss, without a preceding sigmoid. Prediction is positive at `logits >= 0`; diagnostic probabilities use sigmoid.

## Scaling and initialization

`alpha_for_length` implements these exact expressions, using natural logarithms:

```math
\alpha(n)=
\begin{cases}
1 & \text{constant},\\
\log n & \text{log},\\
1+\operatorname{softplus}(k_\alpha)\log(1+n) & \text{learned\_log}.
\end{cases}
\qquad c=\operatorname{softplus}(k_\alpha).
```

The default `alpha_log_scale_init=-5` gives $c\approx0.00671535$. Query/key and head use `nn.Linear` initialization. For input width 2, each weight and head bias is initialized from $\mathcal U(-1/\sqrt2,1/\sqrt2)$; query/key have no bias. Construction order is query, key, classifier, then scalar. `set_reproducibility` runs before model construction.

## Margin, attention and classifier threshold

In the two-token baseline, $a=q_u^\top k_t/\sqrt d$, $b=q_u^\top k_u/\sqrt d$ and raw margin $\Delta=a-b$. The effective pairwise margin is $\alpha(n)\Delta$. Accounting for the number of competitors gives:

```math
g(n)=\alpha(n)\Delta-\log(n-1),\qquad
p_t(n)=\frac{1}{1+(n-1)e^{-\alpha(n)\Delta}}=\sigma(g(n)).
```

For positive head slope $w_t-w_u>0$:

```math
p^*=-\frac{w_u+\beta}{w_t-w_u},\qquad z\ge0\iff p_t\ge p^*.
```

This inequality changes for a negative slope and is undefined at zero slope. Decreasing attention can still support finite-length classification while above the threshold. Under the exact two-score assumptions and fixed trained weights, constant scaling gives $p_t\to0$, log gives $p_t\to1$ when $\Delta>1$, and learned-log gives $p_t\to1$ when $c\Delta>1$.

The condition $c\Delta>1$ characterizes convergence of target mass to 1, not a universal necessary condition for binary classification. At the log boundary $\Delta=1$, the limit is $1/2$; at learned-log's $c\Delta=1$, it is $e^\Delta/(1+e^\Delta)$, so the head matters. Finite 10M accuracy is distinct from infinite-length behavior. These simplified results do not automatically apply to the full Stage 1/2 transformer.

## Registered parameters and active pathways

The baseline registers **12 parameter elements in every scaling mode**, verified with `sum(p.numel() for p in model.parameters())` in the small forward check.

| Component | Registered count | Use in baseline data |
|---|---:|---|
| Query weight `[2,2]` | 4 | Only the 2-element non-target column is queried; target column gradient is zero |
| Key weight `[2,2]` | 4 | Target and non-target keys occur in positives |
| Head weights and bias | 3 | Maps the two value coordinates to a logit |
| `alpha_log_scale_unconstrained` | 1 | Connected to forward only in learned-log |

All have `requires_grad=True` and are passed to AdamW. In constant/log modes, the scalar gradient is `None`, so it is not updated. At baseline `weight_decay=0`, the unused target query column also remains unchanged. Structurally used parameter elements number 9 for constant/log and 10 for learned-log; this is not a count of independent identifiable degrees of freedom. With vocabulary extension, the registered count is $2d(H+M)+4$; the active-column statement above concerns the baseline only.

## Loaders, seeds, epochs and updates

`make_multilength_loaders` creates one fixed `TensorDataset` per length and reuses it throughout training. It does not resample data each epoch. Each batch has one length, without padding. `--train-lengths` overrides `--train-length`, retains list order, and does not deduplicate repeated lengths.

- `train_examples` and `val_examples` are **per length**. With $L$ lengths the training dataset contains $L\times\text{train_examples}$ sequences. `test_examples` is the total at each evaluation length.
- Train dataset seed is `seed + 1 + offset`; shuffled train loader seed is `seed + 10001 + offset`. Validation dataset seed is `seed + 2 + offset`, without loader shuffle.
- Global Python/PyTorch seed is `seed`. Deterministic algorithms are requested with `warn_only=True`; this does not guarantee identical bits across devices or versions.
- `run_loaders_once` materializes a list of all batches, then shuffles that list for training with `random.Random(seed + 20000 + epoch)`. This also applies to single-length training. Each loader's internal permutation changes as its generator advances. Validation uses the unshuffled length-list order.
- `drop_last=False`. With 2000 examples and batch size 64, there are 31 full batches and one 16-example batch: 32 updates per epoch. A mixed epoch has the sum of $\lceil E/B\rceil$ over lengths.
- Without `max_train_steps`, epochs limit training. When specified, the update limit replaces the epoch limit and training repeats datasets until that many updates. The final epoch can stop partway through its batches, followed by full validation. There is no early stopping or best-checkpoint selection; only the final model is saved.
- History loss is weighted by processed example count. A partial batch still consumes an update, so equal steps may not mean equal sequences. Train accuracy aggregates batch predictions made immediately before each update; validation uses the model after the epoch.

Checkpoints contain final model state and partial configuration metadata, without optimizer/RNG state, historical Git revision or intermediate checkpoints. Keep the adjacent `config.json` for full settings.

## Chunked evaluation and metric aggregation

`main` uses `seed + 10000 + n` for evaluation at length $n$. `iter_eval_batches` creates at most `eval_chunk_examples` sequences per chunk, then batches them with `eval_batch_size`. Chunk seeds increase by 1. `chunk_label_counts` preserves global positive/negative counts. Stratified sampling cycles position-bucket/final-query-id/target-id combinations and carries offsets between chunks; the default is random sampling.

**A single sequence is not split into token chunks.** Each example still has its complete `[n]` token ids and forward creates keys `[B,n,d]` and scores/attention `[B,n]`. Memory scales with full length and chunk/batch sizes. Chunking examples reduces memory but does not make 10M evaluation inherently small. This cleanup used lengths 10 and 20 only.

`evaluate_length` retains per-example scalar diagnostics on CPU, rather than all sequences for the entire evaluation set. Overall accuracy aggregates all examples; class accuracy aggregates its class. Target scores, margin, attention and theory errors average positives. `std_non_target_scores` averages each positive sequence's non-target score standard deviation. Test metrics do not include loss; train/val loss is in history. With identical baseline positives, positive accuracy is either 0 or 1.

`add_train_delta_theory` uses the evaluation row for the first training length, falling back to the first evaluation row if absent. Include training lengths in the evaluation list before interpreting the `using_train_delta` column. With multiple non-target types, mean-margin two-score theory is not exact: inspect generalized theory and worst-case margin separately.

## A length-10 calculation

This controlled example is not a trained result. Set $d=2$, $q_u=[1,0]^\top$, $k_t=[2\sqrt2,0]^\top$, $k_u=[0,0]^\top$, and head $w_t=2$, $w_u=-1$, $\beta=0.1$. Then $a=2$, $b=0$, $\Delta=2$, $z=3p_t-0.9$ and $p^*=0.3$.

| Scaling | $\alpha(10)$ | Positive $p_t$ | Positive logit | Negative logit |
|---|---:|---:|---:|---:|
| Constant | 1.000000 | 0.450853 | 0.452559 | -0.900000 |
| Log | 2.302585 | 0.917431 | 1.852294 | -0.900000 |
| Learned-log initialized at -5 | 1.016103 | 0.458839 | 0.476516 | -0.900000 |

Independent Python scalar calculations agreed with actual float32 forward readouts and positive/negative logits at `rtol=2e-6`, `atol=2e-7`. These tolerances allow rounding in this small length-10 reduction, not arbitrary long-sequence error. Forward was not changed, so a new gradient test was unnecessary.

Stage 4A reuses this model, dataset and evaluation; Stage 4B reuses the model and several helpers. No shared code was changed or related stage retrained during cleanup.
