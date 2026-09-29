"""Audit restart equivalence, growth crossings, failure rankings and loss matching."""

import argparse
import csv
import json
import math
import shutil
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_training_length_pilot import attention_gap, predicted_failure_log10
from stage3_simplified_attention import project_dir, write_csv, write_json

CONDITIONS = ["10", "100", "10 100"]
FIELDS = ["c", "delta", "c_delta", "head_slope", "head_intercept", "head_threshold", "balanced_bce"]


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def mean_std(values, digits=3):
    return f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def failure_status(values):
    """Separate invalid heads, finite descending failures and verified no-crossing cases."""
    threshold = values["head_threshold"]
    if values["head_slope"] <= 0 or threshold is None or not 0 < threshold < 1:
        return "head_not_applicable", None
    if values["c_delta"] < 1:
        return "finite", predicted_failure_log10("learned_log", values)
    target_gap = math.log(threshold) - math.log1p(-threshold)
    if values["c_delta"] == 1:
        min_gap = values["delta"]
    else:
        critical_length = max(2, (values["c_delta"] + 1) / (values["c_delta"] - 1))
        min_gap = attention_gap("learned_log", values, math.log(critical_length))[0]
    if min_gap >= target_gap:
        return "no_finite_crossing", None
    return "boundary_or_nonmonotone_threshold_case", None


def match_loss(trajectory, key, target):
    """Linearly interpolate adjacent recorded diagnostics at the first loss crossing."""
    for before, after in zip(trajectory, trajectory[1:]):
        high, low = before[key], after[key]
        if high >= target >= low and high > low:
            fraction = (high - target) / (high - low)
            return {"update_lower": before["optimizer_updates"], "update_upper": after["optimizer_updates"],
                    "estimated_update": before["optimizer_updates"] + fraction * (after["optimizer_updates"] - before["optimizer_updates"]),
                    **{field: before[field] + fraction * (after[field] - before[field])
                       for field in ["c", "delta", "c_delta", "head_threshold"]}}
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    directory = project_dir() / args.run_dir
    status = json.loads((directory / "status.json").read_text())
    assert status["complete"] and status["completed_runs"] == status["planned_runs"] == 9
    summary = read_csv(directory / "pilot_summary.csv")
    assert len(summary) == 9
    analysis = directory / "analysis"
    analysis.mkdir(exist_ok=True)
    trajectories = {}
    crossings, exposure, verification = [], [], []
    for record in summary:
        run = directory / record["run"]
        seed, lengths = int(record["seed"]), record["train_lengths"]
        assert record["alpha_mode"] == "learned_log" and int(record["optimizer_updates"]) == 6400
        prefix = json.loads((run / "pilot_prefix_verification.json").read_text())
        assert prefix["weights_bitwise_equal"] and prefix["initial_state_equal"]
        assert prefix["diagnostic_rows_exactly_equal"] == 36
        assert prefix["train_history_rows_exactly_equal"] == (25 if lengths == "10 100" else 50)
        verification.append({"run": record["run"], **prefix})
        counts = json.loads((run / "exposure.json").read_text())
        assert counts["total_sequences"] == 409600
        for length, entry in counts["by_length"].items():
            expected = 204800 if lengths == "10 100" else 409600
            assert entry["sequences"] == expected
            assert entry["positives"] == entry["negatives"] == expected // 2
            assert entry["tokens"] == expected * int(length)
            exposure.append({"run": record["run"], "length": int(length), **entry})
        raw = read_csv(run / "training_diagnostics.csv")
        steps = sorted({int(row["optimizer_updates"]) for row in raw})
        assert steps == list(range(0, 6401, 200)) and len(raw) == 132
        trajectory = []
        for step in steps:
            rows = {int(row["length"]): row for row in raw if int(row["optimizer_updates"]) == step}
            values = {field: float(rows[100][field]) if rows[100][field] else None for field in FIELDS}
            values["optimizer_updates"] = step
            values["common100_bce"] = float(rows[100]["balanced_bce"])
            values["objective_bce"] = statistics.mean(float(rows[int(length)]["balanced_bce"]) for length in lengths.split())
            values["training_classes_correct"] = all(float(rows[int(length)]["positive_logit"]) >= 0 and float(rows[int(length)]["negative_logit"]) < 0 for length in lengths.split())
            values["failure_status"], values["failure_log10"] = failure_status(values)
            trajectory.append(values)
        trajectories[(seed, lengths)] = trajectory
        first_index = next((index for index, row in enumerate(trajectory) if row["c_delta"] > 1), None)
        sustained = next((index for index, row in enumerate(trajectory)
                          if row["c_delta"] > 1 and all(later["c_delta"] > 1 for later in trajectory[index:])), None)
        crossings.append({"seed": seed, "train_lengths": lengths,
                          "first_above": trajectory[first_index]["optimizer_updates"] if first_index is not None else None,
                          "previous_recording": trajectory[first_index - 1]["optimizer_updates"] if first_index is not None and first_index > 0 else None,
                          "all_later_recordings_above": all(row["c_delta"] > 1 for row in trajectory[first_index:]) if first_index is not None else False,
                          "sustained_above_from_recording": trajectory[sustained]["optimizer_updates"] if sustained is not None else None,
                          "final_c_delta": trajectory[-1]["c_delta"]})
    write_csv(analysis / "crossings.csv", crossings)
    write_csv(analysis / "realized_exposure.csv", exposure)
    write_json(analysis / "prefix_verification.json", verification)

    paired = []
    for seed in range(3):
        for single, mixed in zip(trajectories[(seed, "100")], trajectories[(seed, "10 100")]):
            paired.append({"seed": seed, "optimizer_updates": single["optimizer_updates"],
                           **{field + "_mixed_minus_100": mixed[field] - single[field]
                              for field in ["c", "delta", "c_delta", "head_threshold", "objective_bce", "common100_bce"]}})
    write_csv(analysis / "paired_differences_by_step.csv", paired)
    ranking_rows, ranking_changes = [], []
    for seed in range(3):
        previous = None
        for index in range(33):
            members = [(lengths, trajectories[(seed, lengths)][index]) for lengths in CONDITIONS]
            comparable = all(row["training_classes_correct"] and row["failure_status"] in {"finite", "no_finite_crossing"} for _, row in members)
            ranking = "not_comparable_before_fitting_or_head_validity"
            if comparable:
                ordered = sorted(((lengths, math.inf if row["failure_status"] == "no_finite_crossing" else row["failure_log10"]) for lengths, row in members), key=lambda item: item[1], reverse=True)
                groups = []
                for lengths, value in ordered:
                    if groups and value == groups[-1][0]:
                        groups[-1][1].append(lengths)
                    else:
                        groups.append((value, [lengths]))
                ranking = " > ".join(" = ".join(f"[{name.replace(' ', ',')}]" for name in group) for _, group in groups)
                if ranking != previous:
                    ranking_changes.append({"seed": seed, "optimizer_updates": members[0][1]["optimizer_updates"], "ranking": ranking})
                    previous = ranking
            for lengths, row in members:
                ranking_rows.append({"seed": seed, "train_lengths": lengths, "optimizer_updates": row["optimizer_updates"],
                                     "prediction_status": row["failure_status"], "predicted_failure_log10_length": row["failure_log10"],
                                     "all_conditions_comparable": comparable, "ranking_longest_first": ranking})
    write_csv(analysis / "failure_rankings_by_step.csv", ranking_rows)
    write_csv(analysis / "failure_ranking_changes.csv", ranking_changes)

    matched, matched_differences = [], []
    for loss_key in ["objective_bce", "common100_bce"]:
        for target in [0.04, 0.02, 0.01, 0.005]:
            for seed in range(3):
                matches = {}
                for lengths in CONDITIONS:
                    result = match_loss(trajectories[(seed, lengths)], loss_key, target)
                    if result is not None:
                        matches[lengths] = result
                        matched.append({"loss_definition": loss_key, "target_bce": target, "seed": seed,
                                        "train_lengths": lengths, **result})
                if "100" in matches and "10 100" in matches:
                    single, mixed = matches["100"], matches["10 100"]
                    matched_differences.append({"loss_definition": loss_key, "target_bce": target, "seed": seed,
                                                **{field + "_mixed_minus_100": mixed[field] - single[field]
                                                   for field in ["estimated_update", "c", "delta", "c_delta", "head_threshold"]}})
    write_csv(analysis / "loss_matched_parameters.csv", matched)
    write_csv(analysis / "loss_matched_paired_differences.csv", matched_differences)
    flattened = [{"seed": seed, "train_lengths": lengths, **row} for (seed, lengths), trajectory in trajectories.items() for row in trajectory]
    write_csv(analysis / "trajectory_summary.csv", flattened)

    colors = {"10": "#0072B2", "100": "#D55E00", "10 100": "#009E73"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    keys = ["c", "delta", "c_delta", "objective_bce", "common100_bce", "head_threshold"]
    labels = ["Coefficient c", "Raw margin Δ", "Exponent cΔ", "Own objective BCE", "Common length-100 BCE", "Head threshold"]
    for lengths in CONDITIONS:
        for axis, key, label in zip(axes.flat, keys, labels):
            steps = [row["optimizer_updates"] for row in trajectories[(0, lengths)]]
            means = [statistics.mean(trajectories[(seed, lengths)][index][key] for seed in range(3)) for index in range(33)]
            stds = [statistics.stdev(trajectories[(seed, lengths)][index][key] for seed in range(3)) for index in range(33)]
            axis.plot(steps, means, color=colors[lengths], label=f"[{lengths.replace(' ', ',')}]")
            axis.fill_between(steps, [a - b for a, b in zip(means, stds)], [a + b for a, b in zip(means, stds)], color=colors[lengths], alpha=0.15)
            axis.set(xlabel="Optimizer updates", ylabel=label)
            axis.axvline(1600, color="gray", linestyle=":", linewidth=1)
            axis.grid(alpha=0.2)
    axes[0, 2].axhline(1, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_yscale("log")
    axes[1, 1].set_yscale("log")
    axes[0, 0].legend()
    fig.suptitle("Learned-log fresh restarts: mean ± sample SD, seeds 0/1/2; dotted line = pilot budget")
    fig.tight_layout()
    fig.savefig(analysis / "confirmation_trajectories.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for seed, axis in enumerate(axes):
        for lengths in CONDITIONS:
            rows = [row for row in trajectories[(seed, lengths)] if row["training_classes_correct"] and row["failure_status"] == "finite"]
            axis.plot([row["optimizer_updates"] for row in rows], [row["failure_log10"] for row in rows], color=colors[lengths], label=f"[{lengths.replace(' ', ',')}]")
        axis.set(title=f"Seed {seed}", xlabel="Updates", ylabel="Predicted log10 failure length")
        axis.set_ylim(0, 60)
        axis.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle("Analytical failure predictions; values above 60 clipped, no finite crossing omitted")
    fig.tight_layout()
    fig.savefig(analysis / "failure_length_trajectories.png", dpi=170)
    plt.close(fig)

    tables = []
    for step in [1600, 6400]:
        for lengths in CONDITIONS:
            members = [next(row for row in trajectories[(seed, lengths)] if row["optimizer_updates"] == step) for seed in range(3)]
            tables.append(f"| {step} | [{lengths.replace(' ', ',')}] | {mean_std([row['c'] for row in members], 4)} | {mean_std([row['delta'] for row in members])} | {mean_std([row['c_delta'] for row in members])} | {mean_std([row['objective_bce'] for row in members], 5)} |")
    crossing_table = []
    for row in crossings:
        interval = f"({row['previous_recording']}, {row['first_above']}]" if row["first_above"] else "Not observed by 6400"
        crossing_table.append(f"| {row['seed']} | [{row['train_lengths'].replace(' ', ',')}] | {interval} | {row['all_later_recordings_above']} | {row['final_c_delta']:.4f} |")
    difference_table = []
    for step in [1600, 3200, 4800, 6400]:
        members = [row for row in paired if row["optimizer_updates"] == step]
        difference_table.append(f"| {step} | {mean_std([row['delta_mixed_minus_100'] for row in members])} | {mean_std([row['c_delta_mixed_minus_100'] for row in members])} | " + ", ".join(f"{row['c_delta_mixed_minus_100']:+.4f}" for row in members) + " |")
    matching_table = []
    matched_levels_table = []
    for loss_key in ["objective_bce", "common100_bce"]:
        for target in [0.04, 0.02, 0.01, 0.005]:
            members = [row for row in matched_differences if row["loss_definition"] == loss_key and row["target_bce"] == target]
            if len(members) == 3:
                matching_table.append(f"| {loss_key} | {target} | {mean_std([row['estimated_update_mixed_minus_100'] for row in members], 1)} | {mean_std([row['delta_mixed_minus_100'] for row in members])} | {mean_std([row['c_delta_mixed_minus_100'] for row in members])} |")
            if loss_key == "common100_bce":
                columns = []
                for lengths in CONDITIONS:
                    values = [row["c_delta"] for row in matched if row["loss_definition"] == loss_key
                              and row["target_bce"] == target and row["train_lengths"] == lengths]
                    columns.append(mean_std(values) if len(values) == 3 else "not reached by every seed")
                matched_levels_table.append(f"| {target} | " + " | ".join(columns) + " |")
    ranking_table = [f"| {row['seed']} | {row['optimizer_updates']} | {row['ranking']} |" for row in ranking_changes]
    prefix = "../../" + directory.relative_to(project_dir()).as_posix()
    text = fr"""# Learned-Log 6400-Update Confirmation Results

All nine fresh runs completed in {status['wall_seconds'] / 60:.2f} minutes.
The [launch specification](CONFIRMATION_6400_SPEC.md) fixes the original three
training conditions, seeds and model, changing only the budget to 6400 updates.
All runs restarted from seed initialization with a new optimizer. At 1600 updates,
**all nine** matched pilot weights bit for bit, all 36 diagnostic rows exactly,
and all history rows through that budget exactly. No checkpoint weights were
loaded to continue optimization. See [prefix verification]({prefix}/analysis/prefix_verification.json).
The six related tests passed, including unchanged final weights, RNG and history
with the intermediate verification/checkpoint enabled, independent failure-formula
checks, head-validity checks and loss-interpolation checks.

## Answers to the four questions

1. **The mixture's exponent advantage over `[100]` persists.** It is positive
   in every seed at every recorded point from 1600 through 6400 updates. The paired
   mean difference grows from about 0.015 to 0.026. The mixture still has a smaller
   raw margin, so this is not a simple uniform improvement in attention parameters.
2. **Only `[10]` enters the above-1 regime within this budget.** Seeds 0 and 1
   first appear above 1 at 5600 updates, seed 2 at 4400, and remain above at every
   later recorded point through 6400. Neither `[100]` nor the mixture crosses by 6400.
3. **Finite failure-length rankings reverse before threshold entry.** At 1600,
   all seeds rank `[100] > [10,100] > [10]`; by 6400 all rank `[10] > [10,100] > [100]`.
   Mixture overtakes `[100]` at the first recorded points 2200, 6200 and 2600 for
   seeds 0, 1 and 2 respectively. These are sampled ranking-change times, not exact
   continuous crossing times. The change demonstrates why final exponents alone
   do not describe the entire training trajectory.
4. **A scalar progress-speed explanation is insufficient for the observed gaps.**
   `[100]` reaches shared length-100 loss targets sooner than the mixture, yet has
   a smaller exponent. At matched common length-100 BCE, mixture-minus-100 exponent
   differences remain about 0.028–0.031 while raw-margin differences remain negative.
   This is descriptive evidence of different parameter allocation at comparable
   measured performance, not proof of convergence or a uniquely identified cause.

## Parameter trajectories and the small exponent advantage

Means ± sample standard deviations across the three paired seeds:

| Updates | Training lengths | $c$ | $\Delta$ | $c\Delta$ | Balanced own-objective BCE |
|---|---|---:|---:|---:|---:|
{chr(10).join(tables)}

Mixture minus `[100]` differences (paired across seeds):

| Updates | $\Delta$ difference | $c\Delta$ difference | Per-seed exponent differences (0,1,2) |
|---|---:|---:|---|
{chr(10).join(difference_table)}

![Parameter and loss trajectories]({prefix}/analysis/confirmation_trajectories.png)

The gray vertical line marks 1600 updates. The plots show changing parameters
and both the native training objective and the common length-100 diagnostic.
Analytical balanced BCE uses the exact two-score model; measured batch/validation
losses remain in each run's `train_history.csv`.

## First growth-threshold entry and persistence

Strictly above $c\Delta=1$ is recorded at 200-update intervals. The intervals
bracket a crossing between observations; they do not locate an exact update.
Persistence means all later observations **through 6400**, not indefinitely.

| Seed | Training lengths | First crossing interval | All later observations above 1 | Final $c\Delta$ |
|---|---|---|---|---:|
{chr(10).join(crossing_table)}

## Predicted failure-length rankings

Ranks run from longest predicted failure length to shortest. `>` describes the
head-aware analytical prediction, not measured long-sequence accuracy. A verified
no-finite-threshold-crossing case ranks above finite predictions; tied no-crossing
cases are grouped. Rankings begin only once all three conditions classify their
training classes correctly and their heads support this comparison.

| Seed | First observation of this ranking | Longest predicted failure first |
|---|---:|---|
{chr(10).join(ranking_table)}

![Analytical failure-length trajectories]({prefix}/analysis/failure_length_trajectories.png)

Values beyond $\log_{{10}} n=60$ are clipped only in the plot; the full values
are in [failure rankings]({prefix}/analysis/failure_rankings_by_step.csv).
No-finite-crossing cases are omitted from the finite curves. The solver checks
head slope/threshold and, when the exponent exceeds 1, the minimum attention gap;
an exponent alone is not treated as a universal classification criterion.

## Does matching progress remove the parameter difference?

Match the first downward crossing of a shared loss using linear interpolation
between adjacent 200-update recordings. This is approximate descriptive matching,
not extra training or causal identification. `objective_bce` uses each condition's
own balanced length distribution; `common100_bce` evaluates every model at length
100, so its definition is shared across conditions. Compare both rather than
equating unlike training losses.

| Matched loss | Target BCE | Estimated updates: mixture minus 100 | $\Delta$ difference | $c\Delta$ difference |
|---|---:|---:|---:|---:|
{chr(10).join(matching_table)}

For the common loss definition, the exponent levels across all three conditions are:

| Matched length-100 BCE | `[10]` $c\Delta$ | `[100]` $c\Delta$ | `[10,100]` $c\Delta$ |
|---|---:|---:|---:|
{chr(10).join(matched_levels_table)}

Full interpolation brackets, per-seed parameters and differences are in
[loss-matched parameters]({prefix}/analysis/loss_matched_parameters.csv) and
[paired differences]({prefix}/analysis/loss_matched_paired_differences.csv).
Persistence of a parameter gap at matched loss argues against describing it only
as a scalar training-speed shift; it does not prove a unique mechanism or rule out
all optimization/transient explanations. Loss curves still changing at 6400 limit
claims about stable limiting parameter selection.

## Exposure, explicit evaluation and reproducibility

Every run processed 409600 sequences, balanced by class. Mixtures used 204800
sequences per length. Token totals were 4096000 (`[10]`), 40960000 (`[100]`) and
22528000 (`[10,100]`). Same maximum length and updates do not match token budget
or length-100 exposure. Recorded counts are [here]({prefix}/analysis/realized_exposure.csv).

Final explicit sequence evaluation remained 10, 100, 1000, 10000, 50 examples per
length. All 36 run/length attention checks matched the independent two-score
formula within `rtol=3e-5`, `atol=3e-6`. Beyond that grid, including failure lengths,
values are closed-form predictions; no 10M sequence evaluation was performed.

Each run saves final and step-1600 model-only checkpoints, history, 132 diagnostic
rows (33 times × 4 lengths), exposure and verification JSON. No optimizer state is
saved. [Manifest]({prefix}/manifest.json), `source_snapshot/`,
[run summary]({prefix}/pilot_summary.csv), and [trajectory summary]({prefix}/analysis/trajectory_summary.csv)
preserve settings, code/environment and all seeds. Outputs remain ignored/local;
the original pilot and report artifacts were read only.
"""
    report = project_dir() / "documents/training_length/CONFIRMATION_6400_RESULTS.md"
    report.write_text(text, encoding="utf-8")
    shutil.copy2(__file__, analysis / "analyze_training_length_confirmation.py")
    write_json(analysis / "audit.json", {"runs": 9, "prefix_verification_passed": 9,
                                         "cadence_and_exposure_verified": True, "wall_seconds": status["wall_seconds"]})
    print("\n".join(tables + difference_table + crossing_table + matching_table).replace("±", "+/-"))
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
