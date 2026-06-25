# CLAUDE.md

This file provides concise guidance to Claude Code working in this repository.

## Source Of Truth

Treat `infinite_generalization/documents/` as the source of truth for research intent and experiment design.

## Commands And Setup

- Use the commands in `infinite_generalization/README.md`.
- Use the example configs in `infinite_generalization/configs/`.
- Put local configs in `infinite_generalization/configs/local/`; this directory is ignored except for `.gitkeep`.
- Use one virtual environment at the repository root: `.venv`.

## Platform Notes

- The repo is on Windows: `D:\03_Coding\microLLM-generalization`.
- Two shells are available: a Bash tool (Git Bash, POSIX `sh`) and PowerShell. The README documents the human workflow in PowerShell with an activated venv (`.\.venv\Scripts\Activate.ps1`).
- When running through the Bash tool, the PowerShell activation does not apply. Call the venv interpreter directly and set `PYTHONPATH` inline, e.g. from `infinite_generalization`: `PYTHONPATH=src ../.venv/Scripts/python.exe src/stage3_simplified_attention.py ...`.
- Match the syntax to the shell you actually invoke; do not mix PowerShell-only syntax into Bash commands or vice versa.
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
- Keep commit messages short and focused on the main change.
- Do not write "Co-Authored-By" or some equivalence.

## Research Context

The project studies length generalization in very small models on existential target-token tasks.

- Stage 1: a minimal transformer trained at short length does not learn a true length-invariant solution. Numerical analysis attributes failure to fixed-margin attention dilution plus length-growing non-target/max-pool effects.
- Stage 2B: tested length-aware transformer interventions. Some helped finite extrapolation, but the full-transformer setting stayed harder than the simplified theory.
- Stage 3: moved to a professor-suggested reduced attention model where the final query attends over token embeddings, to test the theory directly in a simpler architecture.

Stage 3 findings so far:
- Constant multiplier fails at long length because target attention dilutes.
- Fixed-log multiplier succeeds when the score margin is large enough.
- Learned-log multiplier can succeed after enough optimization pushes `c * Delta > 1`.
- Stage 3B multi-length training did not reliably help; it tended to increase raw margin `Delta` while keeping learned coefficient `c` low.
- Stage 3C: target position does not materially affect the simplified model.
- Stage 3D: multiple non-target token types make worst-case margin the important diagnostic.
- Stage 3C+D sanity check matched Stage 3D; target-anywhere placement did not change the main conclusion.
- Stage 3E implemented multiple target token types and completed base experiments.

## Key Documents

- `infinite_generalization/documents/TASK.md`
- `infinite_generalization/documents/STAGE3_SIMPLIFIED_LENGTH_AWARE_ATTENTION.md`
- `infinite_generalization/documents/STAGE3_WEIGHT_LEVEL_MECHANISM.md`
- `infinite_generalization/documents/STAGE3C_TARGET_ANYWHERE.md`
- `infinite_generalization/documents/STAGE3CD_TARGET_ANYWHERE_MULTI_NONTARGET.md`
- `infinite_generalization/documents/STAGE3D_MULTIPLE_NON_TARGET_TOKENS.md`
- `infinite_generalization/documents/STAGE3E_MULTIPLE_TARGET_TOKENS.md`

## Important Code Files

- Main Stage 3 experiment code: `infinite_generalization/src/stage3_simplified_attention.py`
- Mechanistic analysis: `infinite_generalization/src/analyze_stage3_mechanism.py`
- Tests: `infinite_generalization/tests/`

The Stage 3E implementation supports multiple target token types via `target_token_count`, multiple non-target token types via `non_target_token_count`, target-anywhere mode via `target_position_mode`, and target-type diagnostics in `target_type_metrics.csv`.

Recent run locations:
- Stage 3C+D sanity check: `infinite_generalization/runs/stage3cd_target_anywhere_multi_nontarget/`
- Stage 3E base runs: `infinite_generalization/runs/stage3e_multiple_targets/`

## Cautions

- Do not overwrite existing user changes. Check `git status --short` before editing.
- Be careful with memory at length 10M. Chunked evaluation exists to avoid creating huge full evaluation tensors.
- Keep the distinction clear: the simplified Stage 3 conclusions do not automatically transfer to the full Stage 1/2 transformer.
- For learned-log success, the key asymptotic diagnostic is whether the effective worst-case `c * Delta` exceeds 1.
