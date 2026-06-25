# AGENTS.md

This file provides concise guidance to coding agents working in this repository.

## Source Of Truth

Treat `infinite_generalization/documents/` as the source of truth for research intent and experiment design.

## Commands And Setup

- Use the commands in `infinite_generalization/README.md`.
- Use the example configs in `infinite_generalization/configs/`.
- Put local configs in `infinite_generalization/configs/local/`; this directory is ignored except for `.gitkeep`.
- Use one virtual environment at the repository root: `.venv`.

## Platform Notes

- The repo is on Windows: `D:\03_Coding\microLLM-generalization`.
- The default shell is PowerShell. Use PowerShell syntax in commands.
- Every code comment must be written in English.

## Document Math Formatting

These rules apply when writing or editing Markdown documents (for example under `infinite_generalization/documents/`).

- Wrap inline math in `$...$`, never in backticks; backticks do not render as math. Examples: `$c\Delta > 1$`, `$\Delta_{\min}$`, `$\alpha$`, `$o(n)$`, `$m_{\text{non}}$`.
- Keep backticks only for non-math code spans: file names, config keys, run names, code identifiers, and library expressions (for example `eval_chunk_examples`, `learned_log_e200_t3_nt1`, `Linear(H+1, H+1)`).
- Leave bare measured or data values as plain text or backticks, not math (for example `1.000`, `0.860`).
- Use fenced code blocks tagged `math` for display equations, and fenced code blocks tagged `text` for token-sequence layouts and config listings; do not convert these blocks to `$...$`.

## Activity Log

- Add an activity log entry only when the user explicitly asks.
- Keep `activitylog.md` in the project root directory.
- Write activity log entries in English.
- Use a title, such as `May 26 Activity Log`, and a short body.
- Keep entries concise unless explicitly asked for detail.

## Commit Guidelines

- Write commit messages in English.
- Keep commit messages focused on the main change.

## Research Context

The project studies length generalization in very small models on existential target-token tasks.

- Stage 1: a minimal transformer trained at short length does not learn a true length-invariant solution. Numerical analysis attributes failure to fixed-margin attention dilution plus length-growing non-target/max-pool effects.
- Stage 2B: tested length-aware transformer interventions. Some helped finite extrapolation, but the full-transformer setting stayed harder than the simplified theory.
- Stage 3: moved to a professor-suggested reduced attention model where the final query attends over token embeddings, to test the theory directly in a simpler architecture.
- Stage 4A: extended the reduced model from a binary present/absent detector to a non-binary identity classifier that names which target token type is present, or a dedicated none class.

Stage 3 findings so far:
- Constant multiplier fails at long length because target attention dilutes.
- Fixed-log multiplier succeeds when the score margin is large enough.
- Learned-log multiplier can succeed after enough optimization pushes `c * Delta > 1`.
- Stage 3B multi-length training did not reliably help; it tended to increase raw margin `Delta` while keeping learned coefficient `c` low.
- Stage 3C: target position does not materially affect the simplified model.
- Stage 3D: multiple non-target token types make worst-case margin the important diagnostic.
- Stage 3C+D sanity check matched Stage 3D; target-anywhere placement did not change the main conclusion.
- Stage 3E implemented multiple target token types and completed base experiments.

Stage 4A findings so far:
- The Stage 3 length-generalization story transfers to non-binary classification: constant fails, fixed-log succeeds, and learned-log succeeds once worst-case `c * Delta` exceeds 1.
- The multi-class failure mode is presence collapse (positives misclassified as the none class), not type confusion; the smallest-margin target type fails first.
- Learned-log at 6400 steps passed the 10M benchmark with worst-case `c * Delta = 0.86 < 1` (asymptotically incomplete); 9600 steps pushed it to `c * Delta = 1.02 > 1`.

## Key Documents

- `infinite_generalization/documents/TASK.md`
- `infinite_generalization/documents/STAGE3_SIMPLIFIED_LENGTH_AWARE_ATTENTION.md`
- `infinite_generalization/documents/STAGE3_WEIGHT_LEVEL_MECHANISM.md`
- `infinite_generalization/documents/STAGE3C_TARGET_ANYWHERE.md`
- `infinite_generalization/documents/STAGE3CD_TARGET_ANYWHERE_MULTI_NONTARGET.md`
- `infinite_generalization/documents/STAGE3D_MULTIPLE_NON_TARGET_TOKENS.md`
- `infinite_generalization/documents/STAGE3E_MULTIPLE_TARGET_TOKENS.md`
- `infinite_generalization/documents/STAGE4A_NONBINARY_CLASSIFICATION.md`

## Important Code Files

- Main Stage 3 experiment code: `infinite_generalization/src/stage3_simplified_attention.py`
- Stage 4A non-binary classification code: `infinite_generalization/src/stage4a_nonbinary_classification.py`
- Mechanistic analysis: `infinite_generalization/src/analyze_stage3_mechanism.py`
- Tests: `infinite_generalization/tests/`

The Stage 3E implementation supports multiple target token types via `target_token_count`, multiple non-target token types via `non_target_token_count`, target-anywhere mode via `target_position_mode`, and target-type diagnostics in `target_type_metrics.csv`.

The Stage 4A implementation reuses the Stage 3 dataset, chunked and stratified evaluation, and length-aware `alpha` modes, adding a multi-class value pathway (per-target-type attention mass), a head over `H + 1` classes (`H` target types plus a none class), and multi-class diagnostics. Its tests live in `infinite_generalization/tests/test_stage4a_nonbinary.py`.

Recent run locations:
- Stage 4A base runs: `infinite_generalization/runs/stage4a/`

## Cautions

- Do not overwrite existing user changes. Check `git status --short` before editing.
- Be careful with memory at length 10M. Chunked evaluation exists to avoid creating huge full evaluation tensors.
- Keep the distinction clear: the simplified Stage 3 conclusions do not automatically transfer to the full Stage 1/2 transformer.
- For learned-log success, the key asymptotic diagnostic is whether the effective worst-case `c * Delta` exceeds 1.
