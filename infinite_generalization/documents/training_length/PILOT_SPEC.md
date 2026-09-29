# Detection Training-Length Pilot: Launch Specification

## Authorization and scope

The updated [PLAN.md](PLAN.md) defines the small contrast `[10]`, `[100]`, `[10,100]`.
On 2026-09-28 the user requested execution and selected a **60-minute total execution
cap**. The launch adopts the plan's suggested numerical defaults below. This is an
exploratory pilot, with no automatic 6400-step extension, length-1000 training or
explicit 10M evaluation.

## Predictions written before launch

Constant scaling should fit training lengths but remain subject to dilution;
training at 100 may increase the raw margin and postpone the predicted failure
relative to training at 10. Fixed-log should supply a useful reference for how
training length changes the learned score margin rather than its fixed multiplier.

For learned-log, the historical Stage 3B result suggests that the mixture may
increase raw margin without increasing the asymptotic exponent relative to the
same-maximum single-length condition. This is a tentative prediction, not a
required outcome. A consistently larger mixture exponent across paired seeds,
or a pattern explained by incomplete fitting instead of calibration, would
contradict or weaken that interpretation. All conditions and seeds will be reported.

The 1600-step budget may leave a condition undertrained. High finite accuracy
does not establish completed optimization, and a lower exponent alone does not
establish a worse classifier. Review loss, head threshold, effective margins,
trajectories and analytical failure predictions together before selecting a follow-up.

## Exact launch settings

| Setting | Value |
|---|---|
| Conditions | 3 training-length sets × 3 scaling modes × 3 seeds = 27 runs |
| Length sets | `[10]`, `[100]`, `[10,100]` |
| Scaling modes | `constant`, `log`, `learned_log` |
| Paired seeds | 0, 1, 2; full initial-state SHA-256 checked across lengths |
| Optimizer updates | 1600 per run; restart from initialization, no early stopping |
| Train dataset | 2048 examples **per length**, 1024 positive and 1024 negative |
| Val dataset | 512 examples per length, balanced |
| Batch size | 64; every training batch full; existing loader and batch shuffle retained |
| Dataset reuse | Existing fixed-dataset reuse |
| Optimizer | AdamW, learning rate 0.003, weight decay 0 |
| Model | Existing two-token detector, fixed-start target, final non-target query, width 2 |
| Learned-log initialization | Unconstrained scalar -5 |
| Explicit evaluation | 10, 100, 1000, 10000; 50 examples each, random mode |
| Eval batching | Chunk examples 50; batch size 16 |
| Diagnostic cadence | Initialization, every 200 updates and final update |
| Device | CUDA, NVIDIA GeForce RTX 4060 Laptop GPU |
| Cap | 60 minutes for the pilot sweep |

2048 examples is a documented pilot change from the report's historical 2000,
which remains preserved in its reproduction specification. At 1600 updates,
single-length runs complete 50 epochs of 32 batches; mixtures complete 25 epochs
of 64 batches. Each run processes exactly **102400 sequences**, balanced by class.
Mixtures receive 51200 sequences at each length. These planned counts will also
be checked against actual observed batches.

Token exposure differs by design: 1024000 tokens for `[10]`, 10240000 for `[100]`,
and 5632000 for `[10,100]`. Equal updates/sequences do not mean equal tokens or
equal training time. Mixture and `[100]` share their maximum length but the
mixture gets half as many length-100 sequences; retain this distinction when
interpreting results.

## Recording and validation

[run_training_length_pilot.py](../../src/run_training_length_pilot.py) calls the
existing Stage 3 `train_model` and evaluation functions. A default-off passive
observer records realized class/length/sequence/token exposure and lightweight
two-score diagnostics. It does not change architecture, optimizer, RNG calls,
batch allocation or default entry-point behavior. Diagnostics include $c$,
$\Delta$, $c\Delta$, head slope/threshold, effective margins, target mass,
positive/negative logits and balanced analytical BCE at each explicit grid length.
History still records measured train/val loss through the existing training path.

The recorder is checked against the unobserved path for identical CPU final
weights, RNG state and history, and for identical CUDA final weights/history
in the timed 200-update mixed-length run. Related Stage 3/4 tests are run before
launch. An independent constant-scaling formula checks the failure-length solver.

Only final model checkpoints are saved. Longer-budget runs, if authorized later,
must restart or add explicit optimizer/RNG checkpoint support; these checkpoints
cannot faithfully resume optimization. Diagnostics beyond the explicit grid,
including 10M and farther lengths, are **closed-form predictions**. Failure
length estimates solve the two-score head threshold and are not measured failures.

## Output and command

Timing and recorder-equivalence artifacts: `runs/training_length/pilot_20260928_01/`.
Full pilot: `runs/training_length/pilot_20260928_01/sweep/`, initially nonexistent.
The manifest records exact settings, current revision/dirty state, environment,
source snapshot and the complete planned run list before training begins.

Run from the repository root in PowerShell:

```powershell
& ./.venv/Scripts/python.exe infinite_generalization/src/run_training_length_pilot.py --output-dir runs/training_length/pilot_20260928_01/sweep --wall-time-minutes 60 --device cuda --steps 1600 --seeds 0 1 2 --diagnostic-interval 200
```

The runner refuses an existing sweep directory, updates `status.json` and
`pilot_summary.csv` after each run, and checks remaining time before launching
another run. The old report and experiment artifacts are read-only.
