# Learned-Log Mechanism Follow-up Launch Specification

Launch date: 2026-09-29. Follow [the fixed plan](MECHANISM_FOLLOWUP_PLAN.md).
Phase 1 uses the unchanged confirmation runner, seeds 3–7, learned-log,
`[10]`, `[100]`, `[10,100]`, 6400 updates, and diagnostics at initialization
and every 200 updates. It uses a new result directory. The reference seeds 0–2
are read from the prior confirmation without retraining.

Before seeing new outcomes, the primary predictions are: `[10]` tends to have
the largest final $c\Delta$; `[10,100]` tends to exceed `[100]` by a small
amount; and `[10]` is more likely to enter $c\Delta>1$ by 6400. Any opposite
sign, recrossing or non-entry is retained, not discarded. Crossing is sampled
at 200-update intervals and right-censored at 6400.

Phase 2 uses the same seed and data pairing and single lengths only, changing
the initial coefficient to 0.1 or 1.0 with the softplus inverse. Baseline cells
are reused from phase 1. A reversal or disappearance of the length ordering is
the predeclared evidence of initialization sensitivity. The intervention also
changes initial attention; no matched-attention claim is made.

Phase 3 traces fresh, paired replays for seeds 3–5, both single lengths, and
baseline/0.1 initial coefficients. A 1.0 trace is added only if 1.0 alone
changes the ordering. Trace windows and the exact update accounting are fixed
in the plan. Replay weights and original diagnostics must match their source
sweeps; no model-only checkpoint is loaded to resume. A failed equivalence check
stops trace interpretation.

All runs retain the model, data construction, AdamW settings, explicit eval
lengths, and no-10M restriction of [the 6400-step specification](CONFIRMATION_6400_SPEC.md).
Use a provisional 60-minute wall-time ceiling across new training. Record full
settings, source snapshots, realized exposure and completion status in each
new output directory. Run the phases sequentially.

Execution note: the predeclared conditional six $c_0=1.0$ traces were activated
because seed 5 failed to fit at this initialization for both single lengths,
whereas it fitted at baseline and 0.1. This note records the decision made
after phase 2, before the full trace; it does not revise the initial predictions.
