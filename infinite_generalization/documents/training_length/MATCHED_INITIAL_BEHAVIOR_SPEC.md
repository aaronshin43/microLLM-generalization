# Matched initial behavior: launch specification

Date: 2026-10-04. This specification implements `MATCHED_INITIAL_BEHAVIOR_PLAN.md`.

## Intervention

Construct each model afresh with the original seed and unchanged head. Let $c_b$ be the actual float32 `softplus(-5)` baseline coefficient and $c_0$ the actual float32 coefficient after setting its unconstrained parameter. Preserve the signed raw gap, including negative-gap initializations.

```math
r = \frac{1+c_b\log(1+n_{\mathrm{ref}})}{1+c_0\log(1+n_{\mathrm{ref}})},\qquad
W_q' = \sqrt{r} W_q,\quad W_k' = \sqrt{r} W_k.
```

Transform both complete projection matrices before creating AdamW or recording update zero. At the reference length, verify corrected scores, attention weights, value outputs, logits, and per-example BCE against the baseline with `rtol=3e-5`, `atol=3e-6`. Verify head equality bitwise, raw-score scaling, direction preservation, baseline initialization hashes, and unchanged random states. Save both initial states and numerical residuals. The zero-change baseline bypass skips multiplication entirely.

Only the reference length's positive behavior is matched; negative logits agree at every length. Raw gaps, norms, $c\Delta$, Jacobians, gradients, and subsequent optimizer paths are allowed to differ. Changing $c_0$ does not permit simultaneous positive-behavior matching at both lengths when the baseline gap is nonzero.

## Runs and comparison

- Main: seeds 3–7, $c_0=0.1,1$, reference 100 with training lengths 10 and 100: 20 runs.
- Own-length supplement: the same seeds and coefficients, reference 10 with training length 10: 10 runs.
- Reuse the ten existing small-coefficient single-length runs and read the twenty existing unmatched coefficient interventions as comparators.
- Verification: two additional baseline replays, seed 3/length 100 and seed 5/length 10, including the initialization bypass and extended observer. Require bitwise equality of final 6400-step weights and exact equality of all diagnostics and history against the existing baseline. Existing checkpoints do not contain separate 1600-step weights; verify the 1600-step diagnostic/history prefix as part of the full-record comparison, without claiming an unavailable intermediate weight comparison.

All trained models start fresh; comparison checkpoints are loaded only into separate comparison objects. Training remains 6400 updates, 2048 balanced training examples, 512 validation examples, batch size 64, AdamW learning rate 0.003 and weight decay zero. Keep the existing model, data generation, sampling, and evaluation settings. Evaluate lengths 10, 100, 1000, and 10000 explicitly. Further-length failure estimates are analytical, not empirical.

## Conditional factorization

Proceed if matching leaves coefficient-dependent fitting outcomes, update trajectories, or final parameter differences. Use seeds 3–5, $c_0=1$, own-length references with training length 10 or 100; add query-only and key-only scaling: 12 runs, compared with six reused symmetric main runs.

```math
(W_q',W_k')=(rW_q,W_k)\quad\text{or}\quad(W_q,rW_k).
```

At a fixed seed, coefficient, and reference, verify these models' initial functions against the symmetric model at all four evaluation lengths. Complete query/key scores are also checked. Their norms and optimization geometry differ despite matching the initial scoring function.

## Recording and budget

Retain 200-update diagnostics and record every actual update from 1–200, 1590–1610, and 3190–3210: 242 traced updates per run. Record coefficient gradients and realized updates, full projection gradients and realized updates, norms and relative update norms, exact midpoint decompositions of $\Delta$, $c\Delta$, and effective margins, positive masses, head parameters, logits, and balanced analytical BCE at lengths 10 and 100. Save source/config/environment snapshots and run manifests.

Use one shared 60-minute training execution ceiling for verification, main, and conditional phases. Stop before launching a run that cannot reasonably fit in the remaining budget. Do not introduce mixture training, length-1000 training, a new task, or an explicit 10M evaluation.
