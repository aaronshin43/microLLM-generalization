# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Repository

 Treat `infinite_generalization/documents/` as the source of truth for what the code, when written, must do.

Key documents (read these before writing code):
- `infinite_generalization/documents/PLAN.md` — overall research plan and staged experiment progression
- `infinite_generalization/documents/TASK.md` — exact spec of the first synthetic task (Token-Presence Detection), including vocab, data generation rules, splits, and evaluation slices
- `infinite_generalization/documents/initial_step.md` — rationale for starting with a 1-layer, ≤2-head, `d_model=64` transformer
- `infinite_generalization/documents/infinite_length_generalization_transformers.md` — background notes

## Research Goal

Study **length generalization** in micro LLMs (<1M params): if a model trains only on short sequences (length 10), does the *same learned computation* extrapolate to lengths 20, 50, 100, 200, 500, 1000? The deeper question is whether the model learns a length-invariant algorithm vs. a length-specific shortcut.

## Architectural Decisions
These are committed in `PLAN.md` and should be respected unless the user explicitly revisits them:

- **Fresh lightweight PyTorch implementation** — *not* nanoGPT. The first task is binary classification, not causal LM, and the project needs full control over pooling, positional encoding, and attention inspection.
- **Stage 0 baseline**: non-transformer max-pooling baseline (Embedding → per-token MLP → max-pool → classifier) before any transformer work. Confirm baseline generalizes nearly perfectly across all eval lengths before touching transformers.
- **Stage 1 transformer**: 1 encoder layer, 1–2 heads, `d_model=64`, **no positional encoding**, max pooling.
- **No length-specific components** in either model: no learned absolute positional embeddings, no sequence flattening, no recurrent state.
- **Evaluation length sweep is mandatory**: every model must be evaluated at lengths `10, 20, 50, 100, 200, 500, 1000` with at least overall / positive-class / negative-class accuracy reported.

## Build / Test Commands

Current Stage 0 commands are listed in the `Stage 0 Commands` section below.

None yet — no `pyproject.toml`, `requirements.txt`, or test runner exists. When implementing Stage 0, set up a minimal Python/PyTorch project and add the relevant commands to this file.

## Platform Notes

- Working tree is on Windows (`D:\03_Coding\microLLM-generalization`); the default shell is PowerShell. Use PowerShell syntax for any commands you suggest (`$env:VAR`, `;` instead of `&&`, etc.).\

**Every comments must be written in English**

## Stage 0 Commands

Smoke test:

```powershell
Set-Location infinite_generalization
$env:PYTHONPATH = "src"
python -m stage0_baseline --smoke-test
```

Unit tests:

```powershell
Set-Location infinite_generalization
python -m unittest discover -s tests
```

Default Stage 0 run:

```powershell
Set-Location infinite_generalization
$env:PYTHONPATH = "src"
python -m stage0_baseline
```

Stage runs write `diagnostic_slices_by_length.csv` with controlled negative, multi-target, and target-position slices. The default diagnostic slice size is `--diagnostic-examples 2000`.

## Stage 1 Commands

Smoke test:

```powershell
Set-Location infinite_generalization
$env:PYTHONPATH = "src"
python -m stage1_transformer --smoke-test
```

Default Stage 1 run:

```powershell
Set-Location infinite_generalization
$env:PYTHONPATH = "src"
python -m stage1_transformer
```

Attention summary run:

```powershell
Set-Location infinite_generalization
$env:PYTHONPATH = "src"
python -m stage1_transformer --save-attention
```

Attention summaries are written per diagnostic slice, not only for the primary balanced evaluation set.

## Activity Log

- Keep `activitylog.md` in the project root directory.
- Write activity log entries in English.
- Use a title (e.g. May 26 Activity Log) and a short body for each entry.
- Keep entries concise; do not include long implementation notes unless explicitly requested.

## Commit Guidelines

- Write commit messages in English.
- Keep commit messages short and focused on the main change.
