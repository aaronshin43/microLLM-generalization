"""Evaluate immutable multiplier-2 forecasts without refitting the response model."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_coefficient_lr_branches import FIELDS, interpolate_loss, md_table
from analyze_training_length_confirmation import failure_status, read_csv
from run_coefficient_lr_holdout import file_digest
from stage3_simplified_attention import project_dir, write_csv, write_json


def load_trajectory(run, length):
    raw = read_csv(run / "training_diagnostics.csv")
    assert len(raw) == 132
    result = []
    for step in range(1600, 6401, 200):
        group = {int(row["length"]): row for row in raw if int(row["optimizer_updates"]) == step}
        common, own = group[100], group[length]
        values = {key: float(common[key]) for key in ("c", "delta", "c_delta", "head_slope", "head_intercept", "head_threshold")}
        values.update(optimizer_updates=step, common100_bce=float(common["balanced_bce"]),
            own_bce=float(own["balanced_bce"]), mass100=float(common["target_mass"]),
            positive100=float(common["positive_logit"]), negative100=float(common["negative_logit"]),
            own_positive=float(own["positive_logit"]), own_negative=float(own["negative_logit"]))
        values["fitting_correct"] = values["own_positive"] > 0 and values["own_negative"] < 0
        values["failure_status"], values["failure_log10"] = failure_status(values)
        result.append(values)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    directory = project_dir() / args.run_dir
    manifest = json.loads((directory / "manifest.json").read_text())
    source = Path(manifest["parent_run"])
    status = json.loads((directory / "status.json").read_text())
    assert status["complete"] and status["completed_branches"] == 6
    frozen_file = directory / "frozen_inputs/predictions.csv"
    assert file_digest(frozen_file) == manifest["prediction_source_sha256"] == file_digest(source / "analysis/held_out_multiplier2_predictions.csv")
    forecasts = read_csv(frozen_file)
    forecast_map = {(int(row["seed"]), int(row["length"]), row["comparison"]): row for row in forecasts}
    analysis = directory / "analysis"
    analysis.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), directory / "source_snapshot" / Path(__file__).name)
    paths, endpoints, matches, allocation, verification, first_updates = {}, [], [], [], [], []
    for seed in (3, 4, 6):
        for length in (10, 100):
            parent = f"n{length}_s{seed}"
            run = directory / "branches" / f"{parent}_x2"
            check = json.loads((run / "branch_verification.json").read_text())
            assert all(check[key] for key in ("initial_state_gradient_and_moments_exact", "first_other_parameter_updates_exact",
                "first_update_proposal_exact", "all_tail_batches_match_control", "parent_and_prediction_files_unchanged", "prefix_history_exact"))
            assert check["traced_updates"] == 232 and check["tail_updates"] == 4800
            verification.append({"seed": seed, "length": length, "all_state_data_checks_passed": True,
                "max_decomposition_residual": check["max_decomposition_residual"]})
            for tag in ("control", "x2", "fast"):
                branch = run if tag == "x2" else source / "branches" / f"{parent}_{tag}"
                paths[(seed, length, tag)] = load_trajectory(branch, length)
            trajectory = paths[(seed, length, "x2")]
            explicit = read_csv(run / "metrics_by_length.csv")
            assert {int(row["length"]) for row in explicit} == {10, 100, 1000, 10000}
            entry = next((i for i, row in enumerate(trajectory) if row["c_delta"] > 1), None)
            endpoints.append({"seed": seed, "length": length, **trajectory[-1],
                "fitting_retained_on_grid": all(row["fitting_correct"] for row in trajectory),
                "explicit_all_classes_correct": all(float(row["positive_accuracy"]) == float(row["negative_accuracy"]) == 1 for row in explicit),
                "first_above": trajectory[entry]["optimizer_updates"] if entry is not None else None,
                "previous_recording": trajectory[entry-1]["optimizer_updates"] if entry is not None else None,
                "all_later_recordings_above": all(row["c_delta"] > 1 for row in trajectory[entry:]) if entry is not None else False})
            exposure = json.loads((run / "exposure.json").read_text())
            assert exposure["total_sequences"] == 409600
            for target in (0.02, 0.01, 0.005):
                values = interpolate_loss(trajectory, "common100_bce", target)
                matches.append({"seed": seed, "length": length, "target_bce": target, "reached": values is not None,
                    **(values or {key: None for key in ("update_lower", "update_upper", "estimated_update", *FIELDS)})})
            old_allocation = json.loads((source / "branches" / f"{parent}_control/branch_verification.json").read_text())["total_allocation"]
            allocation.append({"seed": seed, "length": length, **check["total_allocation"],
                **{key + "_minus_control": check["total_allocation"][key] - old_allocation[key]
                   for key in ("c_part", "score_part", "r_change", "query_gap_part", "key_gap_part")}})
            first = read_csv(run / "update_trace.csv")[0]
            first_updates.append({"seed": seed, "length": length,
                **{key: float(first[key]) for key in ("u_update", "c_update", "delta_update", "c_delta_update", "c_delta_after", "coefficient_lr")}})
    write_csv(analysis / "final_parameters.csv", endpoints)
    write_csv(analysis / "common_loss_parameters.csv", matches)
    write_csv(analysis / "allocation.csv", allocation)
    write_csv(analysis / "verification.csv", verification)
    write_csv(analysis / "first_updates.csv", first_updates)
    write_csv(analysis / "trajectories.csv", [{"seed": seed, "length": length, "tag": tag, **row}
        for (seed, length, tag), trajectory in paths.items() for row in trajectory])
    results, directions, gaps = [], [], []
    compared = {}
    for seed in (3, 4, 6):
        for length in (10, 100):
            for comparison in ("update6400", "common100_bce0.01"):
                cells = {tag: path[-1] if comparison == "update6400" else interpolate_loss(path, "common100_bce", 0.01)
                    for tag in ("control", "x2", "fast") for path in (paths[(seed, length, tag)],)}
                available = all(cells.values())
                compared[(seed, length, comparison)] = cells
                prediction = forecast_map[(seed, length, comparison)]
                if cells["control"] is not None:
                    assert cells["control"]["c_delta"] == float(prediction["control_r"])
                actual_r = cells["x2"]["c_delta"] if cells["x2"] else None
                actual_effect = actual_r - float(prediction["control_r"]) if actual_r is not None else None
                numeric = float(prediction["effect_lower"]) <= actual_effect <= float(prediction["effect_upper"]) if actual_effect is not None else None
                results.append({"seed": seed, "length": length, "comparison": comparison,
                    "reached": actual_r is not None, "control_r": float(prediction["control_r"]),
                    "predicted_r": float(prediction["predicted_r"]), "actual_r": actual_r,
                    "predicted_effect": float(prediction["predicted_effect"]), "actual_effect": actual_effect,
                    "effect_lower": float(prediction["effect_lower"]), "effect_upper": float(prediction["effect_upper"]),
                    "relative_effect_error": actual_effect / float(prediction["predicted_effect"]) - 1 if actual_effect is not None else None,
                    "numeric_passed": numeric,
                    "ordering_passed": cells["control"]["c_delta"] < actual_r < cells["fast"]["c_delta"] if available else None,
                    "estimated_update": cells["x2"].get("estimated_update", 6400) if cells["x2"] else None})
        for comparison in ("update6400", "common100_bce0.01"):
            a, b = compared[(seed, 10, comparison)], compared[(seed, 100, comparison)]
            available = all(a.values()) and all(b.values())
            effects = {length: cells["x2"]["c_delta"] - cells["control"]["c_delta"] if available else None
                for length, cells in ((10, a), (100, b))}
            gap = {tag: a[tag]["c_delta"] - b[tag]["c_delta"] if available else None for tag in ("control", "x2", "fast")}
            directions.append({"seed": seed, "comparison": comparison, "evaluable": available,
                "effect10": effects[10], "effect100": effects[100],
                "larger_effect_at10_passed": effects[10] > effects[100] > 0 if available else None,
                "gap_ordering_passed": gap["control"] < gap["x2"] < gap["fast"] if available else None})
            branch_gap = paths[(seed, 10, "control")][0]["c_delta"] - paths[(seed, 100, "control")][0]["c_delta"]
            gaps.append({"seed": seed, "comparison": comparison, "gap_at1600": branch_gap,
                **{"G_" + tag: value for tag, value in gap.items()},
                "gap_x2_minus_control": gap["x2"] - gap["control"] if available else None})
    write_csv(analysis / "numeric_predictions.csv", results)
    write_csv(analysis / "directional_predictions.csv", directions)
    write_csv(analysis / "length_gaps.csv", gaps)
    numeric_passes = sum(row["numeric_passed"] is True for row in results)
    order_passes = sum(row["ordering_passed"] is True for row in results)
    contrast_passes = sum(row["larger_effect_at10_passed"] is True for row in directions)
    gap_passes = sum(row["gap_ordering_passed"] is True for row in directions)
    audit = {"completed_branches": 6, "forecasts_unchanged": True, "validity_checks_passed": True,
        "numeric_passes": numeric_passes, "numeric_predictions": 12, "ordering_passes": order_passes,
        "length_effect_passes": contrast_passes, "gap_ordering_passes": gap_passes,
        "censored_numeric_predictions": sum(not row["reached"] for row in results),
        "all_fitting_retained": all(row["fitting_retained_on_grid"] for row in endpoints),
        "all_explicit_classes_correct": all(row["explicit_all_classes_correct"] for row in endpoints),
        "max_absolute_relative_effect_error": max(abs(row["relative_effect_error"]) for row in results if row["relative_effect_error"] is not None),
        "max_decomposition_residual": max(row["max_decomposition_residual"] for row in verification),
        "wall_seconds": status["wall_seconds"]}
    audit["positive_effect_errors"] = sum(row["relative_effect_error"] is not None and row["relative_effect_error"] > 0 for row in results)
    predictive_success = (numeric_passes == order_passes == 12 and contrast_passes == gap_passes == 6
                          and audit["all_fitting_retained"])
    audit["overall_predictive_success"] = predictive_success
    audit["selected_next_direction"] = "independent_initialization_validation" if predictive_success else "inspect_response_model_and_compensation"
    excluded = project_dir() / "runs/training_length/coefficient_lr_multiplier2_20261010_01"
    excluded_updates = 0
    if excluded != directory and (excluded / "status.json").exists():
        excluded_status = json.loads((excluded / "status.json").read_text())
        if excluded_status.get("invalid_for_scientific_comparison"):
            excluded_updates = excluded_status["materialized_tail_updates"]
            audit["excluded_attempt"] = {"run": str(excluded), "tail_updates": excluded_updates,
                "reason": excluded_status["reason"], "excluded_from_prediction_scoring": True}
    write_json(analysis / "audit.json", audit)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for comparison, axis in zip(("update6400", "common100_bce0.01"), axes):
        subset = [row for row in results if row["comparison"] == comparison]
        axis.axhspan(-20, 20, color="#009E73", alpha=0.12, label="Frozen ±20% criterion")
        for i, row in enumerate(subset):
            if row["relative_effect_error"] is not None:
                axis.scatter(i, 100 * row["relative_effect_error"], color="#0072B2" if row["length"] == 10 else "#D55E00", s=55)
        axis.axhline(0, color="gray", linewidth=0.8)
        axis.set_xticks(range(6), [f"s{row['seed']}\nL{row['length']}" for row in subset])
        axis.set(title=comparison, ylabel="Error in predicted treatment effect (%)")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=9)
    fig.suptitle("Held-out coefficient LR ×2: fixed predictions, no response-model refit")
    fig.tight_layout()
    fig.savefig(analysis / "frozen_prediction_errors.png", dpi=170)
    plt.close(fig)
    prefix = "../../" + directory.relative_to(project_dir()).as_posix()
    tables = {}
    for comparison in ("update6400", "common100_bce0.01"):
        tables[comparison] = md_table(["Seed", "Train length", "Predicted $r_2$", "Observed $r_2$", "Predicted effect", "Observed effect", "Effect error", "±20% passed"], [
            [row["seed"], row["length"], f"{row['predicted_r']:.6f}", f"{row['actual_r']:.6f}" if row["reached"] else "Censored",
             f"{row['predicted_effect']:.6f}", f"{row['actual_effect']:.6f}" if row["reached"] else "Censored",
             f"{100*row['relative_effect_error']:+.2f}%" if row["reached"] else "Censored", row["numeric_passed"]]
            for row in results if row["comparison"] == comparison])
    direction_table = md_table(["Seed", "Comparison", "$A_{10}(2)$", "$A_{100}(2)$", "Larger effect at 10", "$G_1<G_2<G_3$"], [
        [row["seed"], row["comparison"], f"{row['effect10']:.6f}" if row["evaluable"] else "Censored",
         f"{row['effect100']:.6f}" if row["evaluable"] else "Censored", row["larger_effect_at10_passed"], row["gap_ordering_passed"]] for row in directions])
    final_table = md_table(["Seed", "Train length", "$c$", r"$\Delta$", "$r_2$", "First $r>1$ bracket", "Later samples above", "Analytical failure"], [
        [row["seed"], row["length"], f"{row['c']:.6f}", f"{row['delta']:.4f}", f"{row['c_delta']:.6f}",
         f"({row['previous_recording']}, {row['first_above']}]" if row["first_above"] else "Not observed",
         row["all_later_recordings_above"], f"log10 length {row['failure_log10']:.4f}" if row["failure_status"] == "finite" else row["failure_status"]] for row in endpoints])
    allocation_table = md_table(["Seed", "Train length", "Coefficient effect", "Score effect", "Total exponent effect"], [
        [row["seed"], row["length"], f"{row['c_part_minus_control']:+.6f}", f"{row['score_part_minus_control']:+.6f}", f"{row['r_change_minus_control']:+.6f}"] for row in allocation])
    next_direction = ("The next scoped validation should use an independent initialization with the same task and lengths, "
        "freezing predictions before inspecting its intervention outcome. A new seed is needed to distinguish transfer "
        "of the post-fitting explanation from interpolation within these six parents. Model diversity remains a later decision."
        if predictive_success else "Inspect the response model and recorded compensating head/query/key updates before "
        "adding another intervention. Preserve every numeric or directional miss and do not widen the stored tolerance.")
    text = fr"""# Held-out coefficient multiplier 2 results

Six continuations completed through total update 6400 in {status['wall_seconds']/60:.2f}
minutes. Each restores the same successful 1600-update model/AdamW/data/RNG parent
used in the [three-rate experiment](COEFFICIENT_LR_BRANCH_RESULTS.md). Only the
unconstrained coefficient parameter learning rate is 0.006; query/key and head
remain at 0.003. Seeds: 3, 4, 6. Training lengths: 10, 100.
See the [launch specification](COEFFICIENT_LR_MULTIPLIER2_SPEC.md).

## Frozen predictive test

The original forecast CSV was copied and SHA-256 checked before any full tail.
Its response-model parameters, predictions and ±20% effect tolerance were not
recomputed. Numeric success refers to the effect $r_2-r_1$, not relative error
in the absolute exponent. The original forecast file remains unchanged, including
its historical unexecuted marker; this report records the completed test separately.

- Numerical effect predictions passed: {numeric_passes}/12, with
  {audit['censored_numeric_predictions']} censored predictions.
- Predicted $r_1<r_2<r_3$ ordering passed: {order_passes}/12 comparisons.
- Larger positive effect at length 10 passed: {contrast_passes}/6 comparisons.
- Predicted $G_1<G_2<G_3$ ordering passed: {gap_passes}/6 comparisons.
- Fitting retained on every recorded post-branch own-length point:
  {audit['all_fitting_retained']}.

No model refit or tolerance adjustment was made after observing the new rate.
The largest absolute relative effect error was
{100*audit['max_absolute_relative_effect_error']:.2f}%.
Observed effects exceed the point forecast in {audit['positive_effect_errors']}/12
comparisons. Any systematic underprediction matters even when all stored bounds
pass; success within the tolerance does not imply an unbiased response model.

### Equal total updates: 6400

{tables['update6400']}

### Equal common length-100 balanced BCE: 0.01

{tables['common100_bce0.01']}

Common-loss values are interpolated at the first post-1600 downward crossing
between adjacent 200-update records. Bounds, estimated crossing updates, raw
gap, coefficient, head state and attention diagnostics are saved in
[common_loss_parameters.csv]({prefix}/analysis/common_loss_parameters.csv),
including planned companion targets 0.02 and 0.005. Scalar loss matching does
not equalize geometry, moments or the complete output state.

![Frozen numeric prediction errors]({prefix}/analysis/frozen_prediction_errors.png)

## Length-specific response and update allocation

{direction_table}

The held-out rate tests the previously predicted response of the **post-fitting**
length contrast. These parents already had different geometry and optimizer
histories at 1600; this experiment does not explain the complete earlier gap.
Full control/2/3 trajectories are in
[trajectories.csv]({prefix}/analysis/trajectories.csv), and length gaps in
[length_gaps.csv]({prefix}/analysis/length_gaps.csv).

The exponent-effect allocation below subtracts the original multiplier-1
control's signed all-update totals from each new continuation:

{allocation_table}

This exact midpoint accounting separates coefficient and score terms in
$\delta r=\overline\Delta\,\delta c+\overline c\,\delta\Delta$.
Raw gradients are not substituted for actual updates. The maximum numerical
decomposition residual is {audit['max_decomposition_residual']:.3g}. Double
reconstruction of float32 parameters can differ slightly from ordinary diagnostic
rounding. Detailed 232-update traces, head/projection norms, compact allocation
over all 4800 updates and first-step Adam moments are retained for each path.
The decomposition identity is not itself a causal mediation test.

## Fitting and extrapolation

{final_table}

Explicit positive and negative accuracy is 1.0 for all six paths at lengths
10, 100, 1000 and 10000: {audit['all_explicit_classes_correct']}. Each length
uses the original 50 balanced examples. No explicit evaluation above 10000
was performed. Finite failure values are head-aware analytical log10 length
predictions. `no_finite_crossing` is a statement under the reduced two-score
model's exact formula, not an empirical test at all lengths. Entry brackets
and retention refer to the 200-update recording grid.

## Validity and compute

All six source parent files and the original forecast file remain unchanged.
The continuation implementation hashes, Torch version and GPU match the
previous validated run. Every restored parent starts at epoch 51 / update 1600,
with original AdamW counters and moments. Six disposable first-update proposals
were saved before the full tails and reproduced the actual first updates exactly.
First states, gradients, moments, and query/key/head updates match the historical
multiplier-1 sibling exactly. Every rolling tail batch hash matches that sibling,
and the entire 50-epoch prefix history matches. No control was redefined.

Compute is 28800 valid new scientific updates, 1843200 new sequences, plus six
disposable first-update checks. Prefix training was reused without replay.
Every complete path still represents 409600 balanced sequences. The run stayed
within the inherited 60-minute execution ceiling. Five focused state-restoration,
optimizer-isolation, trace-passivity and multiplier-2 proposal tests passed;
shared training code was not changed for this experiment.

The first execution attempt failed the historical first-gradient identity gate
because deterministic CUDA kernel selection had not been enabled in the new
process. RNG restoration alone does not restore that process flag. The largest
observed key-gradient difference was 1.4551915228366852e-11. The original
`set_reproducibility` settings were then applied before parent restoration, and
all six paths were restarted into the current directory. The incomplete first
attempt is preserved and explicitly excluded from prediction scoring; it adds
{excluded_updates} discarded tail updates and six disposable proposal updates
to actual compute. No forecast was changed in response to this implementation
failure. The valid attempt passes the original exact identity gates.

## Interpretation and next direction

This is a held-out learning-rate test on the same fitted parents. Numeric
prediction success strengthens the local response model only where the stored
bounds passed; preserve any miss above even if directions succeeded. The
power-response fit remains phenomenological and does not establish a universal
optimization law or independent-initialization replication.

No further training was run. Review the full prediction score before expanding
the task or model assumptions. {next_direction}
"""
    (project_dir() / "documents/training_length/COEFFICIENT_LR_MULTIPLIER2_RESULTS.md").write_text(text, encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
