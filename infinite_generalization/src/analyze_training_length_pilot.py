"""Audit and summarize every run in a completed detection pilot."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stage3_simplified_attention import project_dir, write_csv, write_json


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def mean_std(values, digits=3):
    return f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = project_dir() / args.run_dir
    status = json.loads((run_dir / "status.json").read_text())
    assert status["complete"] and status["completed_runs"] == 27
    rows = read_csv(run_dir / "pilot_summary.csv")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert len(rows) == 27
    assert all(float(row["negative_accuracy_10000"]) == 1 for row in rows)
    analysis = run_dir / "analysis"
    analysis.mkdir(exist_ok=True)
    grouped = defaultdict(list)
    by_seed_hash = defaultdict(set)
    exposure_rows = []
    for row in rows:
        directory = run_dir / row["run"]
        exposure = json.loads((directory / "exposure.json").read_text())
        lengths = [int(value) for value in row["train_lengths"].split()]
        assert int(row["optimizer_updates"]) == 1600
        assert exposure["total_sequences"] == 102400
        expected_per_length = 102400 // len(lengths)
        for length in lengths:
            entry = exposure["by_length"][str(length)]
            assert entry["sequences"] == expected_per_length
            assert entry["positives"] == entry["negatives"] == expected_per_length // 2
            assert entry["tokens"] == expected_per_length * length
            exposure_rows.append({"run": row["run"], "length": length, **entry})
        assert exposure["total_tokens"] == sum(expected_per_length * length for length in lengths)
        diagnostics = read_csv(directory / "training_diagnostics.csv")
        assert {int(value["optimizer_updates"]) for value in diagnostics} == set(range(0, 1601, 200))
        assert len(diagnostics) == 36
        explicit = read_csv(directory / "metrics_by_length.csv")
        assert [int(value["length"]) for value in explicit] == [10, 100, 1000, 10000]
        assert all(int(value["positive_examples"]) == int(value["negative_examples"]) == 25 for value in explicit)
        training_rows = [value for value in explicit if int(value["length"]) in lengths]
        row["all_training_classes_correct"] = all(float(value["positive_accuracy"]) == float(value["negative_accuracy"]) == 1 for value in training_rows)
        by_seed_hash[int(row["seed"])].add(row["initial_state_sha256"])
        grouped[(row["alpha_mode"], row["train_lengths"])].append(row)
    assert all(len(values) == 1 for values in by_seed_hash.values())
    write_csv(analysis / "realized_exposure.csv", exposure_rows)
    aggregated = []
    table = []
    for mode in ["constant", "log", "learned_log"]:
        for lengths in ["10", "100", "10 100"]:
            members = grouped[(mode, lengths)]
            entry = {"alpha_mode": mode, "train_lengths": lengths}
            for key in ["delta", "c", "c_delta", "head_threshold", "val_loss", "target_mass_10000", "positive_logit_10000", "wall_seconds"]:
                values = [float(row[key]) for row in members if row[key] != ""]
                entry[key + "_mean"] = statistics.mean(values) if values else None
                entry[key + "_std"] = statistics.stdev(values) if values else None
            entry["positive_passes_10000"] = sum(float(row["positive_accuracy_10000"]) == 1 for row in members)
            entry["negative_passes_10000"] = sum(float(row["negative_accuracy_10000"]) == 1 for row in members)
            entry["training_passes"] = sum(row["all_training_classes_correct"] for row in members)
            failures = [float(row["predicted_failure_log10_length"]) for row in members if row["predicted_failure_log10_length"]]
            failure_text = f"{min(failures):.2f}–{max(failures):.2f}" if failures else "n/a"
            entry["failure_log10_min"] = min(failures) if failures else None
            entry["failure_log10_max"] = max(failures) if failures else None
            aggregated.append(entry)
            c_text = mean_std([float(row["c"]) for row in members], 4) if mode == "learned_log" else "n/a"
            cd_text = mean_std([float(row["c_delta"]) for row in members]) if mode == "learned_log" else "n/a"
            table.append(f"| {mode} | [{lengths.replace(' ', ',')}] | {mean_std([float(row['val_loss']) for row in members], 4)} | {mean_std([float(row['delta']) for row in members])} | {c_text} | {cd_text} | {mean_std([float(row['head_threshold']) for row in members])} | {entry['positive_passes_10000']}/3 | {failure_text} |")
    write_csv(analysis / "condition_summary.csv", aggregated)
    differences = []
    for mode in ["constant", "log", "learned_log"]:
        for seed in range(3):
            single = next(row for row in grouped[(mode, "100")] if int(row["seed"]) == seed)
            mixed = next(row for row in grouped[(mode, "10 100")] if int(row["seed"]) == seed)
            entry = {"alpha_mode": mode, "seed": seed}
            for key in ["delta", "c", "c_delta", "head_threshold", "val_loss", "target_mass_10000"]:
                entry[key + "_mixed_minus_single100"] = float(mixed[key]) - float(single[key]) if mixed[key] else None
            differences.append(entry)
    write_csv(analysis / "paired_differences.csv", differences)

    colors = {"10": "#0072B2", "100": "#D55E00", "10 100": "#009E73"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for lengths in ["10", "100", "10 100"]:
        histories = []
        for row in grouped[("learned_log", lengths)]:
            values = [value for value in read_csv(run_dir / row["run"] / "training_diagnostics.csv") if int(value["length"]) == 10]
            histories.append(values)
        steps = [int(value["optimizer_updates"]) for value in histories[0]]
        for axis, key, label in zip(axes, ["c", "delta", "c_delta"], ["Learned coefficient c", "Raw margin Δ", "Exponent cΔ"]):
            means, stds = [], []
            for index in range(len(steps)):
                values = [float(history[index][key]) for history in histories]
                means.append(statistics.mean(values))
                stds.append(statistics.stdev(values))
            axis.plot(steps, means, color=colors[lengths], label=f"[{lengths.replace(' ', ',')}]")
            axis.fill_between(steps, [mean - std for mean, std in zip(means, stds)],
                              [mean + std for mean, std in zip(means, stds)], color=colors[lengths], alpha=0.15)
            axis.set(xlabel="Optimizer updates", ylabel=label)
            axis.grid(alpha=0.2)
    axes[2].axhline(1, color="black", linestyle="--", linewidth=1, label="Target-mass growth threshold")
    axes[0].legend()
    axes[2].legend(fontsize=8)
    fig.suptitle("Learned-log pilot: mean ± sample standard deviation, three paired seeds")
    fig.tight_layout()
    fig.savefig(analysis / "learned_log_trajectories.png", dpi=180)
    plt.close(fig)

    learned_pairs = [row for row in differences if row["alpha_mode"] == "learned_log"]
    delta_diff = [row["delta_mixed_minus_single100"] for row in learned_pairs]
    c_diff = [row["c_mixed_minus_single100"] for row in learned_pairs]
    exponent_diff = [row["c_delta_mixed_minus_single100"] for row in learned_pairs]
    headline = (f"For learned-log, mixture minus single-length 100 changes raw margin by {mean_std(delta_diff)}, "
                f"coefficient by {mean_std(c_diff, 4)}, and exponent by {mean_std(exponent_diff)}. "
                "These are paired differences across the three seeds, not independent-group estimates.")
    report = project_dir() / "documents/training_length/PILOT_RESULTS.md"
    seed_table = []
    for row in rows:
        cd = f"{float(row['c_delta']):.4f}" if row["c_delta"] else "n/a"
        failure = f"{float(row['predicted_failure_log10_length']):.2f}" if row["predicted_failure_log10_length"] else "n/a"
        seed_table.append(f"| {row['run']} | {float(row['val_loss']):.4f} | {float(row['delta']):.4f} | {cd} | {float(row['head_threshold']):.4f} | {int(float(row['positive_accuracy_10000']))} | {failure} |")
    all_fit = sum(row["all_training_classes_correct"] for row in rows)
    prefix = "../../" + run_dir.relative_to(project_dir()).as_posix()
    text = fr"""# Detection Training-Length Pilot Results

All **27 runs completed** at 1600 optimizer updates each. Sweep wall time was
{status['wall_seconds']:.1f} seconds ({status['wall_seconds']/60:.2f} minutes), below the user's
60-minute cap. Exact settings and predictions written before launch are in
[PILOT_SPEC.md](PILOT_SPEC.md); the user's research plan remains in [PLAN.md](PLAN.md).

## Main comparison

{headline}

{all_fit}/27 runs classified both classes correctly at all of their training lengths.
This is a fitting check, not evidence that optimization has converged. All seed-level
conditions are reported below; no longer-budget confirmation has been performed.

Values are means ± sample standard deviations across seeds 0, 1, 2. `Positive passes`
counts seeds with measured positive accuracy 1 at length 10000. Failure ranges give
$\log_{{10}} n^*$ from a **closed-form prediction**, not explicitly evaluated failure lengths.
An `n/a` entry means the descending finite threshold solver does not apply; inspect
the head and scaling regime rather than treating this as unconditional success.

| Mode | Training lengths | Final val BCE | $\Delta$ | $c$ | $c\Delta$ | Head threshold | Positive passes at 10000 | Predicted $\log_{{10}} n^*$ range |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

All final negative predictions at length 10000 were correct in all 27 runs.
For fixed-log the effective growth exponent is $\Delta$, not a learned coefficient.
Constant scaling always has asymptotic target mass 0 for fixed finite weights.
For learned-log, assess $c\Delta$ together with the head; passing a finite benchmark
does not imply convergence of target mass to 1.

![Learned-log optimization trajectories]({prefix}/analysis/learned_log_trajectories.png)

The trajectories use exact two-score analytical diagnostics at initialization and
every 200 updates. Shading is one sample standard deviation across three seeds.
Training/validation losses in each run's history are measured through the original
training loop; analytical diagnostic BCE is labeled separately.

## Controls and realized exposure

### Interpretation against the pre-launch prediction

The same-maximum comparison did **not** show the tentative pattern of a mixture
raising raw margin while lowering the exponent relative to `[100]`. In each paired
seed, the mixture had a smaller $\Delta$, larger $c$, and slightly larger $c\Delta$.
Its analytical failure length was nevertheless **earlier** than `[100]` in every seed:
the smaller constant margin and head threshold also matter at finite lengths.
This is a useful warning against ranking trained classifiers by $c\Delta$ alone.

Relative to `[10]`, the learned-log mixture had a larger raw margin but a smaller
coefficient and exponent in each seed, consistent with the direction of the older
Stage 3B comparison. The choice of single-length reference changes the observation;
the same-maximum comparison is essential for interpreting a mixture.

All nine learned-log runs have $c\Delta<1$ and positive head thresholds, so their
exact two-score analytical predictions imply eventual positive-class collapse even
though they pass the explicit grid. All fixed-log runs have $\Delta>1$ and heads
that distinguish the limiting positive/negative representations. These conclusions
are scoped to fixed trained parameters in this reduced architecture.

The losses and trajectories are still changing at the end of 1600 updates. A
matched longer-budget comparison is needed before calling the observed parameter
selection stable. The paired exponent increase versus `[100]` is small, and three
seeds cannot support a broad claim about multi-length training.

### Exposure audit

Every run processed exactly 102400 sequences, 51200 positive and 51200 negative.
Mixtures processed 51200 sequences at each length, balanced within each length.
Token totals per run were 1024000 (`[10]`), 10240000 (`[100]`) and 5632000 (`[10,100]`).
Full initial state hashes match across every condition for each paired seed,
including across modes. Model, optimizer, learning rate and dataset reuse were fixed.
The mixture has half the length-100 exposure of `[100]`, so it isolates neither
token budget nor length-100 sample count; account for this in mechanism claims.

Explicit sequence evaluation used lengths 10, 100, 1000 and 10000, with 25 positive
and 25 negative examples at each length. Actual forward target masses agreed with
independent two-score calculations within `rtol=3e-5`, `atol=3e-6` in all 108
run/length comparisons. This tolerance permits float32 reduction rounding at the
modest explicit lengths and is not a 10M validation claim.

Beyond this grid, `analytical_predictions.csv` contains closed-form predictions
through $10^{{30}}$. No 10M sequence was generated or measured. Failure estimates
use the trained head threshold, assuming the exact two-token model and positive
head slope. Intermediate optimizer states were not saved; future longer-budget
comparisons must restart unless faithful resume support is added.

## Per-seed results

Positive accuracy is measured at length 10000; predicted failure is $\log_{{10}} n^*$.

| Run | Final val BCE | $\Delta$ | $c\Delta$ | Head threshold | Positive accuracy | Predicted failure |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(seed_table)}

## Artifacts and validation

- [Run manifest and environment]({prefix}/manifest.json): exact configs, revision,
  dirty state, device and launch arguments; `source_snapshot/` preserves the code used.
- [All 27 run summaries]({prefix}/pilot_summary.csv), [condition summary]({prefix}/analysis/condition_summary.csv)
  and [paired differences]({prefix}/analysis/paired_differences.csv).
- [Realized exposure]({prefix}/analysis/realized_exposure.csv) and each run's
  `exposure.json`, `training_diagnostics.csv`, `train_history.csv`, `model.pt`,
  `metrics_by_length.csv` and diagnostic CSVs.
- New observer/failure-solver tests: 2 passed; existing Stage 3 tests: 11 passed;
  Stage 4 tests: 34 passed after the single temporary-directory environment failure
  was rerun with workspace storage. The original Stage 4A test failure was a write
  permission error, not a model assertion failure. CPU observation preserved weights,
  RNG state and history; a 200-update CUDA comparison preserved final weights/history.

Current launch revision: `{manifest['environment']['revision']}`, with recorded source
changes in the manifest/snapshot. Python {manifest['environment']['python']}, PyTorch
{manifest['environment']['torch']}, device `{manifest['environment']['device']}`,
GPU {manifest['environment']['gpu']}. Existing historical artifacts were preserved.
Run outputs remain ignored under the repository policy, so these artifact links
are local and need explicit archiving for use on another machine.

## Review before another launch

This three-seed, one-budget pilot is exploratory. Compare paired effects and their
trajectories with final loss before selecting a 6400-update matched confirmation.
Mixture advantages confined to short finite benchmarks, a lower coefficient by
itself, or training accuracy alone do not establish a new generalization mechanism.
Confirm a selected pattern with matched longer training and additional seeds before
expanding lengths or claiming novelty. No confirmation or intervention was launched.
"""
    report.write_text(text, encoding="utf-8")
    write_json(analysis / "audit.json", {"runs": 27, "initial_pairing_verified": True,
                                         "exposure_verified": True, "diagnostic_cadence_verified": True,
                                         "all_training_classes_correct_runs": all_fit,
                                         "wall_seconds": status["wall_seconds"]})
    print(headline.replace("±", "+/-"))
    print("\n".join(table).replace("±", "+/-").replace("–", " to "))
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
