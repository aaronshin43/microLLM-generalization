# Independent-seed coefficient-rate validation

Execution date: 2026-10-10. The user authorized seeds 8 and 9, training lengths
10 and 100, branches at 1600 and coefficient learning-rate multipliers 1 and 2,
ending at 6400. Four shared prefixes produce eight comparison paths. Preserve
the original small coefficient initialization, task, head, query/key learning
rates, AdamW settings, datasets and diagnostic schedule. No 1/3 or 3 multiplier
is run on the new seeds. The inherited execution ceiling is 60 minutes.

## Rule fixed from existing results

Use a new normalized-effect rule, distinct from the previous power-response
model. For old seeds 3, 4 and 6 at each length and comparison:

```math
q_{s,L,t}=\frac{r_{s,L,2}(t)-r_{s,L,1}(t)}{r_{s,L,1}(t)-r_{s,L}(1600)},
\qquad \overline q_{L,t}=\frac{1}{3}\sum_{s\in\{3,4,6\}}q_{s,L,t},
\qquad \widehat A_{s,L}(2,t)=\overline q_{L,t}
[r_{s,L,1}(t)-r_{s,L}(1600)].
```

The predicted exponent is the new control exponent plus this predicted effect.
Calibration uses the validated original three-rate experiment and multiplier-2
run `_02`; the invalid `_01` attempt is excluded. Freeze source SHA-256 hashes,
all twelve old normalized effects and the following four coefficients **before
any new-seed training**:

| Training length | Comparison | Frozen mean normalized effect |
|---:|---|---:|
| 10 | Total update 6400, primary | 0.7521685749951815 |
| 100 | Total update 6400, primary | 0.6264930352686643 |
| 10 | Common length-100 balanced BCE 0.01, secondary | 0.8575632252231385 |
| 100 | Common length-100 balanced BCE 0.01, secondary | 0.6842067538150549 |

Separate comparison coefficients account for the declared evaluation stage;
none is refitted per new seed. Complete all four multiplier-1 controls first.
Use only their results and branch-point exponents to save eight numerical
predictions and their bounds before starting **any** multiplier-2 tail.
This is a conditional predictive test given a new model's control growth, not
a forecast of that control's entire optimization trajectory from initialization.

## Success criteria fixed before intervention

- Primary numeric test: each of the four eligible pairs at update 6400 has
  observed effect within ±20% of its stored predicted effect. The secondary
  common-BCE comparison uses the same ±20% criterion on four additional forecasts.
  Apply tolerance to the effect, not to the absolute exponent. This is a declared
  falsification threshold, not a confidence interval from three old seeds.
- Direction: positive exponent effect at each length; the effect at length 10
  exceeds that at length 100 for each new seed, at equal updates and common loss.
- Retain own-length fitting at every post-branch 200-update recording and final
  direct evaluations. Numeric success alone cannot satisfy the overall claim.
- No new seed replacement, per-model calibration, rate expansion, post-result
  tolerance adjustment, or extension beyond 6400.

At 1600 require both own-length classes strictly correct, positive head slope
and positive raw margin. Record every intended parent's eligibility. If a parent
fails, retain its two planned paths as descriptive diagnostics, clearly exclude
them from the post-fitting numerical claim, and count the failed eligibility
against full four-parent confirmation. Do not silently change seed or branch
time. Censor common-loss targets that are not reached after the branch; do not
extrapolate. Nonpositive control growth is inapplicable to this positive-growth
rule and must be reported, not fitted around.

## State, measurements, and scope

Capture full model/AdamW/data/RNG/history/exposure at the end of epoch 50 validation.
Enable the original deterministic-kernel settings before every restore. Verify
paired initialization hashes across lengths, 32-update original/split-group
identity checks, independent restored storage, sibling first gradients/moments
and query/key/head updates, all 4800 future batch hashes and prefix histories.
Reuse the already tested serialized continuation code without optimizer restart.

Analytical diagnostics: every 200 global updates. Detailed traces: parent
1590–1600, tails 1601–1800, 3190–3210, 6390–6400 (232 tail records). Keep actual
coefficient/query/key/head updates, Adam moments, norms, head state, attention,
and exact compact allocation over all tail updates. Save final continuation
states and explicit evaluation at 10, 100, 1000 and 10000; longer-length failure
is analytical only. Common-loss parameters are interpolated on the 200-update
grid. Record 0.02 and 0.005 companion losses without selecting new primary targets.

Scientific compute: 44800 updates (four times 1600 plus eight times 4800),
2867200 sequence exposures, counting each shared prefix once and each tail.
Sibling tails deliberately reuse batches. Disposable optimizer-group checks add 256 updates.
No old-seed training or new-rate family is included. After this validation,
use the recorded data to analyze why length 10 responds more strongly: geometry,
coefficient parameterization, Adam moments and attenuation by score/head updates.
Do not automatically launch diversity or further-rate experiments.

## Command

From the repository root:

```powershell
.\.venv\Scripts\python.exe infinite_generalization/src/run_coefficient_lr_newseeds.py --output-dir runs/training_length/coefficient_lr_newseeds_20261010_01 --device cuda --wall-time-minutes 60
```

## Completed execution

All eight paths completed in 4.40 minutes. See the
[results](COEFFICIENT_LR_NEWSEEDS_RESULTS.md). The run retains the original
pre-launch specification in its source snapshot, alongside the frozen rule and
numerical forecast hashes; this completion note does not change those artifacts.
