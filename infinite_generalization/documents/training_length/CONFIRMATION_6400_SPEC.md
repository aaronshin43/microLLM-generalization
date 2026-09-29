# Learned-Log 6400-Update Confirmation

The user authorized nine fresh runs: lengths `[10]`, `[100]`, `[10,100]`, seeds
0, 1, 2, learned-log only, 6400 optimizer updates, initial/every-200-update diagnostics.
All other settings match [PILOT_SPEC.md](PILOT_SPEC.md): 2048 balanced train examples
and 512 val examples per length, batch 64, AdamW 0.003 with no weight decay, scalar
initialization -5, fixed datasets and the existing two-token detector. CUDA and
the previously authorized 60-minute cap are retained. No length-1000 training or
task changes are included. Explicit evaluation remains 10, 100, 1000, 10000.

Runs start from the original seeds, with new optimizer states. The old pilot
checkpoint is read only for comparison, never used to initialize or resume learning.
At 1600 updates, save a model-only checkpoint and require bitwise equality with
the pilot's weights, exact diagnostic-prefix equality and initial-state hash equality.
After 6400 updates, also compare the complete history prefix through 1600 updates.
Record checks per run; a mismatch stops the experiment instead of being hidden.

Questions written before launch:

- Does the small mixture-minus-100 exponent advantage persist or reverse?
- Which condition first exceeds $c\Delta=1$, and does it remain above at all later
  recorded points? Crossings are bracketed by 200-update observations, not exact
  update-time claims; no observation before 6400 establishes permanence afterward.
- How do head-aware analytical failure-length rankings change during training?
- Can parameter ordering be reduced to different progress speeds? Compare trajectories
  at equal updates and at matched balanced objective losses. Mixed loss averages
  length-10 and length-100 BCE, so also match the common length-100 BCE; loss matching
  is descriptive and does not by itself identify a causal optimization mechanism.

Expected pattern: single length 10 may cross earlier, as suggested by the pilot's
larger exponent near 1600 updates. Mixture versus 100 is intentionally unresolved;
both a stable advantage and a reversal are reportable. Longer-budget training may
change finite failure ranking before or after target-mass growth crosses its boundary.
Report all seeds and distinguish fitting speed from stable selection evidence.

Each run processes 409600 sequences. Mixtures split those equally by length;
class counts are balanced. Token totals are 4096000, 40960000 and 22528000 for
the three respective conditions. Verify these from actual observed batches.
No optimizer states are saved for resumed training. Beyond the explicit grid,
failure lengths and long-length outputs remain closed-form predictions.

Output: `runs/training_length/confirmation_6400_20260928_01/` (new directory).

```powershell
& ./.venv/Scripts/python.exe infinite_generalization/src/run_training_length_pilot.py --output-dir runs/training_length/confirmation_6400_20260928_01 --wall-time-minutes 60 --device cuda --steps 6400 --seeds 0 1 2 --modes learned_log --diagnostic-interval 200 --reference-pilot runs/training_length/pilot_20260928_01/sweep --verify-step 1600
```
