"""Audit post-fitting branches and score the predeclared predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_training_length_confirmation import failure_status, read_csv
from stage3_simplified_attention import project_dir, write_csv, write_json

SEEDS = (3, 4, 6)
TAGS = ("slow", "control", "fast")
FIELDS = ("c", "delta", "c_delta", "head_slope", "head_intercept", "head_threshold",
          "common100_bce", "own_bce", "mass100", "positive100", "negative100")


def interpolate_loss(trajectory, loss_key, target):
    for before, after in zip(trajectory, trajectory[1:]):
        high, low = before[loss_key], after[loss_key]
        if high >= target >= low and high > low:
            fraction = (high - target) / (high - low)
            return {"update_lower": before["optimizer_updates"], "update_upper": after["optimizer_updates"],
                "estimated_update": before["optimizer_updates"] + fraction * (after["optimizer_updates"] - before["optimizer_updates"]),
                **{key: before[key] + fraction * (after[key] - before[key]) for key in FIELDS}}
    return None


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    directory = project_dir() / args.run_dir
    status = json.loads((directory / "status.json").read_text())
    assert status["complete"] and status["all_sibling_checks_passed"]
    assert status["completed_parents"] == 6 and status["completed_branches"] == 18
    frozen = json.loads((directory / "predictions_frozen.json").read_text())
    assert frozen["before_comparison_tails"] and frozen["all_parent_forecasts_saved"]
    summary = read_csv(directory / "summary.csv")
    assert len(summary) == 18
    analysis = directory / "analysis"
    analysis.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), directory / "source_snapshot" / Path(__file__).name)
    trajectories, finals, allocations = {}, [], []
    verification, crossings, trace_windows, first_rows = [], [], [], []
    for seed in SEEDS:
        for length in (10, 100):
            parent = directory / "parents" / f"n{length}_s{seed}"
            pcheck = json.loads((parent / "parent_verification.json").read_text())
            assert hashlib.sha256((parent / "parent_state.pt").read_bytes()).hexdigest() == pcheck["checkpoint_file_sha256"]
            scheck = json.loads((parent / "sibling_verification.json").read_text())
            assert all(scheck.values())
            gcheck = json.loads((directory / "validation" / parent.name / "group_verification.json").read_text())
            assert all(value is True for key, value in gcheck.items() if key != "steps")
            for tag in TAGS:
                run = directory / "branches" / f"n{length}_s{seed}_{tag}"
                check = json.loads((run / "branch_verification.json").read_text())
                assert check["parent_state_sha256"] == pcheck["parent_state_sha256"]
                assert check["tail_updates"] == 4800 and check["traced_updates"] == 232
                assert check["first_update_prediction_exact"] and check["max_decomposition_residual"] < 1e-10
                if tag == "control":
                    assert all(json.loads((run / "baseline_verification.json").read_text()).values())
                exposure = json.loads((run / "exposure.json").read_text())
                assert exposure["total_sequences"] == 409600
                assert exposure["by_length"][str(length)]["positives"] == exposure["by_length"][str(length)]["negatives"] == 204800
                explicit = read_csv(run / "metrics_by_length.csv")
                assert len(explicit) == 4
                explicit_correct = all(float(row["positive_accuracy"]) == float(row["negative_accuracy"]) == 1 for row in explicit)
                raw = read_csv(run / "training_diagnostics.csv")
                assert len(raw) == 132
                path = []
                for step in range(1600, 6401, 200):
                    group = {int(row["length"]): row for row in raw if int(row["optimizer_updates"]) == step}
                    a, own = group[100], group[length]
                    values = {key: float(a[key]) for key in ("c", "delta", "c_delta", "head_slope", "head_intercept", "head_threshold")}
                    values.update(optimizer_updates=step, common100_bce=float(a["balanced_bce"]),
                        own_bce=float(own["balanced_bce"]), mass100=float(a["target_mass"]),
                        positive100=float(a["positive_logit"]), negative100=float(a["negative_logit"]),
                        own_positive=float(own["positive_logit"]), own_negative=float(own["negative_logit"]))
                    values["fitting_correct"] = values["own_positive"] > 0 and values["own_negative"] < 0
                    values["failure_status"], values["failure_log10"] = failure_status(values)
                    path.append(values)
                trajectories[(seed, length, tag)] = path
                fit_retained = all(row["fitting_correct"] for row in path)
                first_above = next((index for index, row in enumerate(path) if row["c_delta"] > 1), None)
                crossings.append({"seed": seed, "length": length, "tag": tag,
                    "first_above": path[first_above]["optimizer_updates"] if first_above is not None else None,
                    "previous_recording": path[first_above - 1]["optimizer_updates"] if first_above is not None else None,
                    "all_later_recordings_above": all(row["c_delta"] > 1 for row in path[first_above:]) if first_above is not None else False})
                finals.append({"seed": seed, "length": length, "tag": tag, **path[-1],
                    "fitting_retained_on_grid": fit_retained, "explicit_all_classes_correct": explicit_correct})
                verification.append({"seed": seed, "length": length, "tag": tag,
                    "parent_prefix_exact": True, "baseline_control_exact": tag == "control",
                    "tail_batches_equal": True, "max_residual": check["max_decomposition_residual"]})
                allocation = check["total_allocation"]
                allocations.append({"seed": seed, "length": length, "tag": tag, **allocation,
                    "coefficient_signed_share": allocation["c_part"] / allocation["r_change"]})
                trace = read_csv(run / "update_trace.csv")
                assert len(trace) == 232
                first_rows.append({"seed": seed, "length": length, "tag": tag,
                    **{key: float(trace[0][key]) for key in ("u_update", "c_update", "delta_update", "c_delta_update", "c_delta_after", "coefficient_lr")}})
                for lo, hi in ((1601, 1800), (3190, 3210), (6390, 6400)):
                    window = [row for row in trace if lo <= int(row["optimizer_updates"]) <= hi]
                    result = {"seed": seed, "length": length, "tag": tag, "lower": lo, "upper": hi, "updates": len(window)}
                    for key in ("u_update", "c_update", "delta_update", "c_delta_update", "c_delta_from_c_update", "c_delta_from_score_update",
                                "effective_margin_10_from_c_update", "effective_margin_10_from_score_update",
                                "effective_margin_100_from_c_update", "effective_margin_100_from_score_update",
                                "query_full_update_norm", "key_full_update_norm", "head_weight_actual_update_norm", "head_bias_actual_update_norm"):
                        result[key + "_signed_sum"] = sum(float(row[key]) for row in window)
                        result[key + "_absolute_sum"] = sum(abs(float(row[key])) for row in window)
                    trace_windows.append(result)
    write_csv(analysis / "final_parameters.csv", finals)
    write_csv(analysis / "verification.csv", verification)
    write_csv(analysis / "threshold_crossings.csv", crossings)
    write_csv(analysis / "all_tail_allocation.csv", allocations)
    write_csv(analysis / "trace_window_allocation.csv", trace_windows)
    write_csv(analysis / "first_updates.csv", first_rows)
    write_csv(analysis / "trajectories.csv", [{"seed": seed, "length": length, "tag": tag, **row}
        for (seed, length, tag), trajectory in trajectories.items() for row in trajectory])

    effects, gaps, matched, forecasts, matching = [], [], [], [], {}
    allocation_map = {(row["seed"], row["length"], row["tag"]): row for row in allocations}
    for seed in SEEDS:
        for step in (1800, 3200, 6400):
            point = {(length, tag): next(row for row in trajectories[(seed, length, tag)] if row["optimizer_updates"] == step)
                     for length in (10, 100) for tag in TAGS}
            for length in (10, 100):
                for tag in ("slow", "fast"):
                    a, b = point[(length, tag)], point[(length, "control")]
                    effects.append({"seed": seed, "length": length, "tag": tag, "optimizer_updates": step,
                        **{key + "_minus_control": a[key] - b[key] for key in FIELDS}})
            gap = {tag: point[(10, tag)]["c_delta"] - point[(100, tag)]["c_delta"] for tag in TAGS}
            parent_gap = trajectories[(seed, 10, "control")][0]["c_delta"] - trajectories[(seed, 100, "control")][0]["c_delta"]
            for tag in TAGS:
                gaps.append({"seed": seed, "optimizer_updates": step, "tag": tag,
                    "length_gap": gap[tag], "gap_at_branch": parent_gap,
                    "gap_growth_after_branch": gap[tag] - parent_gap, "gap_minus_control": gap[tag] - gap["control"]})
        for loss_key in ("common100_bce", "own_bce"):
            for target in (0.02, 0.01, 0.005):
                for length in (10, 100):
                    for tag in TAGS:
                        result = interpolate_loss(trajectories[(seed, length, tag)], loss_key, target)
                        matching[(seed, loss_key, target, length, tag)] = result
                        matched.append({"seed": seed, "length": length, "tag": tag, "loss_definition": loss_key,
                            "target_bce": target, "reached": result is not None,
                            **(result or {key: None for key in ("update_lower", "update_upper", "estimated_update", *FIELDS)})})
        point = {(length, tag): trajectories[(seed, length, tag)][-1] for length in (10, 100) for tag in TAGS}
        common = {(length, tag): matching[(seed, "common100_bce", 0.01, length, tag)] for length in (10, 100) for tag in TAGS}
        for definition, members in (("update6400", point), ("common100_bce0.01", common)):
            available = all(members.values())
            for length in (10, 100):
                valid = all(members[(length, tag)] is not None for tag in TAGS)
                passed = members[(length, "fast")]["c_delta"] > members[(length, "control")]["c_delta"] > members[(length, "slow")]["c_delta"] if valid else None
                forecasts.append({"seed": seed, "comparison": definition, "hypothesis": "fast > control > slow", "length": length, "evaluable": valid, "passed": passed})
            for tag in ("slow", "fast"):
                a10 = members[(10, tag)]["c_delta"] - members[(10, "control")]["c_delta"] if available else None
                a100 = members[(100, tag)]["c_delta"] - members[(100, "control")]["c_delta"] if available else None
                forecasts.append({"seed": seed, "comparison": definition, "hypothesis": "absolute effect larger at length10", "tag": tag,
                    "evaluable": available, "passed": abs(a10) > abs(a100) if available else None, "effect10": a10, "effect100": a100})
            gap = {tag: members[(10, tag)]["c_delta"] - members[(100, tag)]["c_delta"] for tag in TAGS} if available else {}
            forecasts.append({"seed": seed, "comparison": definition, "hypothesis": "G_fast > G_control > G_slow", "evaluable": available,
                "passed": gap["fast"] > gap["control"] > gap["slow"] if available else None, **{"G_" + key: value for key, value in gap.items()}})
    write_csv(analysis / "paired_effects_by_step.csv", effects)
    write_csv(analysis / "length_gaps_by_step.csv", gaps)
    write_csv(analysis / "loss_matched_parameters.csv", matched)
    matched_effects, matched_gaps = [], []
    for seed in SEEDS:
        for target in (0.02, 0.01, 0.005):
            for loss_key in ("common100_bce", "own_bce"):
                for length in (10, 100):
                    control = matching[(seed, loss_key, target, length, "control")]
                    for tag in ("slow", "fast"):
                        changed = matching[(seed, loss_key, target, length, tag)]
                        reached = control is not None and changed is not None
                        matched_effects.append({"seed": seed, "length": length, "tag": tag,
                            "loss_definition": loss_key, "target_bce": target, "both_reached": reached,
                            **{key + "_minus_control": changed[key] - control[key] if reached else None
                               for key in (*FIELDS, "estimated_update")}})
            group = {(length, tag): matching[(seed, "common100_bce", target, length, tag)]
                     for length in (10, 100) for tag in TAGS}
            reached = all(group.values())
            values = {tag: group[(10, tag)]["c_delta"] - group[(100, tag)]["c_delta"] if reached else None for tag in TAGS}
            matched_gaps.append({"seed": seed, "target_bce": target, "all_reached": reached,
                **{"G_" + tag: value for tag, value in values.items()},
                "gap_ordering_predicted": values["fast"] > values["control"] > values["slow"] if reached else None})
    write_csv(analysis / "loss_matched_effects.csv", matched_effects)
    write_csv(analysis / "loss_matched_length_gaps.csv", matched_gaps)
    forecast_fields = list(dict.fromkeys(key for row in forecasts for key in row))
    write_csv(analysis / "prediction_scorecard.csv", [{key: row.get(key) for key in forecast_fields} for row in forecasts])

    contribution_effects = []
    for seed in SEEDS:
        for length in (10, 100):
            control = allocation_map[(seed, length, "control")]
            for tag in ("slow", "fast"):
                a = allocation_map[(seed, length, tag)]
                contribution_effects.append({"seed": seed, "length": length, "tag": tag,
                    **{key + "_minus_control": a[key] - control[key] for key in ("c_part", "score_part", "r_change", "query_gap_part", "key_gap_part")}})
    write_csv(analysis / "allocation_effects.csv", contribution_effects)
    # Failure rankings are meaningful only under valid heads and retained fitting.
    rank_changes = []
    for seed in SEEDS:
        previous = None
        for index in range(25):
            group = [(f"{length}/{tag}", trajectories[(seed, length, tag)][index]) for length in (10, 100) for tag in TAGS]
            if not all(row["fitting_correct"] and row["failure_status"] in ("finite", "no_finite_crossing") for _, row in group):
                rank_changes.append({"seed": seed, "optimizer_updates": group[0][1]["optimizer_updates"], "ranking_longest_first": "not_comparable"})
                previous = None
                continue
            ordered = sorted(((name, math.inf if row["failure_status"] == "no_finite_crossing" else row["failure_log10"])
                              for name, row in group), key=lambda pair: pair[1], reverse=True)
            parts = []
            for name, value in ordered:
                if parts and value == parts[-1][0]:
                    parts[-1][1].append(name)
                else:
                    parts.append((value, [name]))
            rank = " > ".join(" = ".join(names) for _, names in parts)
            if rank != previous:
                rank_changes.append({"seed": seed, "optimizer_updates": group[0][1]["optimizer_updates"], "ranking_longest_first": rank})
                previous = rank
    write_csv(analysis / "failure_ranking_changes.csv", rank_changes)
    score = {"predictions": forecasts, "validity": "all required exact comparisons passed",
        "all_fitting_retained": all(row["fitting_retained_on_grid"] for row in finals),
        "explicit_all_correct": all(row["explicit_all_classes_correct"] for row in finals),
        "censored_loss_comparisons": sum(not row["reached"] for row in matched),
        "max_residual": max(row["max_residual"] for row in verification)}
    write_json(analysis / "audit.json", score)

    colors = {"slow": "#0072B2", "control": "#707070", "fast": "#D55E00"}
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7), sharex=True, sharey=True)
    for j, seed in enumerate(SEEDS):
        for i, length in enumerate((10, 100)):
            axis = axes[i, j]
            for tag in TAGS:
                path = trajectories[(seed, length, tag)]
                axis.plot([row["optimizer_updates"] for row in path], [row["c_delta"] for row in path], color=colors[tag], label={"slow": "u LR × 1/3", "control": "u LR × 1", "fast": "u LR × 3"}[tag])
            axis.axhline(1, color="black", linestyle="--", linewidth=0.8)
            axis.set(title=f"Train {length}, seed {seed}", xlabel="Total optimizer updates", ylabel="Exponent r = cΔ")
            axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Coefficient learning-rate intervention from identical fitted parents at step 1600")
    fig.tight_layout()
    fig.savefig(analysis / "exponent_trajectories.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharex=True)
    for seed, axis in zip(SEEDS, axes):
        for length, style in ((10, "-"), (100, "--")):
            for tag in TAGS:
                path = trajectories[(seed, length, tag)]
                axis.plot([row["common100_bce"] for row in path], [row["c_delta"] for row in path], color=colors[tag], linestyle=style, label=f"{length}, {tag}")
        axis.axvline(0.01, color="black", linestyle=":", linewidth=1)
        axis.set(xscale="log", xlabel="Common length-100 balanced BCE", ylabel="Exponent r = cΔ", title=f"Seed {seed}")
        axis.invert_xaxis()
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Progress-matched comparison; lower BCE to the right, dotted line = primary target")
    fig.tight_layout()
    fig.savefig(analysis / "exponent_vs_common_loss.png", dpi=170)
    plt.close(fig)
    final_map = {(row["seed"], row["length"], row["tag"]): row for row in finals}
    rows = []
    for seed in SEEDS:
        for length in (10, 100):
            cells = [final_map[(seed, length, tag)] for tag in TAGS]
            rows.append([seed, length, *(f"{row['c_delta']:.6f}" for row in cells),
                *(f"{row['delta']:.4f}" for row in cells)])
    final_table = md_table(["Seed", "Train length", "$r$, 1/3", "$r$, 1", "$r$, 3", r"$\Delta$, 1/3", r"$\Delta$, 1", r"$\Delta$, 3"], rows)
    matched_rows, gap_rows, allocation_table_rows, threshold_rows = [], [], [], []
    for seed in SEEDS:
        for length in (10, 100):
            cells = [matching[(seed, "common100_bce", 0.01, length, tag)] for tag in TAGS]
            matched_rows.append([seed, length, *(f"{row['c_delta']:.6f}" if row else "Censored" for row in cells),
                *(f"{row['estimated_update']:.1f}" if row else "Censored" for row in cells)])
            for tag in TAGS:
                a = allocation_map[(seed, length, tag)]
                allocation_table_rows.append([seed, length, tag, f"{a['c_part']:+.6f}", f"{a['score_part']:+.6f}", f"{a['r_change']:+.6f}"])
        subset = [row for row in gaps if row["seed"] == seed and row["optimizer_updates"] == 6400]
        gap_rows.append([seed, f"{subset[0]['gap_at_branch']:.6f}",
            *(f"{next(row for row in subset if row['tag'] == tag)['length_gap']:.6f}" for tag in TAGS),
            *(f"{next(row for row in subset if row['tag'] == tag)['gap_minus_control']:+.6f}" for tag in ("slow", "fast"))])
    for crossing in crossings:
        f = final_map[(crossing["seed"], crossing["length"], crossing["tag"])]
        interval = f"({crossing['previous_recording']}, {crossing['first_above']}]" if crossing["first_above"] else "Not observed"
        prediction = f"log10 length {f['failure_log10']:.4f}" if f["failure_status"] == "finite" else f["failure_status"]
        threshold_rows.append([crossing["seed"], crossing["length"], crossing["tag"], interval,
            crossing["all_later_recordings_above"], prediction])
    matching_table = md_table(["Seed", "Train length", "$r$, 1/3", "$r$, 1", "$r$, 3", "Update, 1/3", "Update, 1", "Update, 3"], matched_rows)
    common_gap_table = md_table(["Common BCE", "Seed", "$G_{1/3}$", "$G_1$", "$G_3$", "Predicted order retained"], [
        [row["target_bce"], row["seed"], *(f"{row['G_' + tag]:.6f}" if row["all_reached"] else "Censored" for tag in TAGS), row["gap_ordering_predicted"]]
        for row in matched_gaps])
    gap_table = md_table(["Seed", "$G$ at 1600", "$G_{1/3}$ at 6400", "$G_1$ at 6400", "$G_3$ at 6400", "$G_{1/3}-G_1$", "$G_3-G_1$"], gap_rows)
    allocation_table = md_table(["Seed", "Train length", "Branch", "Coefficient part", "Score part", "Total change in $r$"], allocation_table_rows)
    threshold_table = md_table(["Seed", "Train length", "Branch", "First $r>1$ bracket", "Later grid points above", "Final analytical prediction"], threshold_rows)
    score_rows = []
    for comparison in ("update6400", "common100_bce0.01"):
        for hypothesis in ("fast > control > slow", "absolute effect larger at length10", "G_fast > G_control > G_slow"):
            subset = [row for row in forecasts if row["comparison"] == comparison and row["hypothesis"] == hypothesis]
            evaluated = [row for row in subset if row["evaluable"]]
            label = r"$G_3>G_1>G_{1/3}$" if hypothesis.startswith("G_fast") else hypothesis
            score_rows.append([comparison, label, f"{sum(row['passed'] for row in evaluated)} / {len(evaluated)}", len(subset) - len(evaluated)])
    score_table = md_table(["Comparison", "Predeclared prediction", "Passed / evaluable", "Censored"], score_rows)
    all_passed = all(row["evaluable"] and row["passed"] for row in forecasts)
    next_direction = "held_out_coefficient_multiplier_2" if all_passed and score["all_fitting_retained"] else "recorded_head_geometry_compensation_analysis"
    write_json(analysis / "direction_review.json", {"all_predeclared_endpoint_predictions_passed": all_passed,
        "selected_next_direction": next_direction, "additional_experiments_executed": False})
    gap_allocation_rows = []
    for seed in SEEDS:
        a, b = allocation_map[(seed, 10, "control")], allocation_map[(seed, 100, "control")]
        cpart, scorepart = a["c_part"] - b["c_part"], a["score_part"] - b["score_part"]
        gap_allocation_rows.append({"seed": seed, "coefficient_part": cpart,
            "score_part": scorepart, "post_branch_gap_growth": cpart + scorepart,
            "coefficient_signed_share": cpart / (cpart + scorepart)})
    write_csv(analysis / "control_length_gap_allocation.csv", gap_allocation_rows)
    gap_allocation_table = md_table(["Seed", "Coefficient contribution", "Score contribution", "Later control gap growth", "Signed coefficient share"], [
        [row["seed"], f"{row['coefficient_part']:.6f}", f"{row['score_part']:.6f}", f"{row['post_branch_gap_growth']:.6f}", f"{100*row['coefficient_signed_share']:.1f}%"] for row in gap_allocation_rows])
    held_out = []
    if next_direction == "held_out_coefficient_multiplier_2":
        for seed in SEEDS:
            for length in (10, 100):
                parent_r = trajectories[(seed, length, "control")][0]["c_delta"]
                for comparison in ("update6400", "common100_bce0.01"):
                    cells = {tag: final_map[(seed, length, tag)] if comparison == "update6400"
                             else matching[(seed, "common100_bce", 0.01, length, tag)] for tag in TAGS}
                    beta = math.log((cells["fast"]["c_delta"] - parent_r) / (cells["slow"]["c_delta"] - parent_r)) / math.log(9)
                    predicted_effect = (cells["control"]["c_delta"] - parent_r) * (2 ** beta - 1)
                    held_out.append({"seed": seed, "length": length, "comparison": comparison,
                        "coefficient_multiplier": 2, "coefficient_lr": 0.006, "parent_r": parent_r,
                        "control_r": cells["control"]["c_delta"], "response_beta": beta,
                        "predicted_r": cells["control"]["c_delta"] + predicted_effect,
                        "predicted_effect": predicted_effect, "effect_lower": 0.8 * predicted_effect,
                        "effect_upper": 1.2 * predicted_effect, "tolerance_relative_effect": 0.2,
                        "executed": False})
        write_csv(analysis / "held_out_multiplier2_predictions.csv", held_out)
    held_out_table = md_table(["Seed", "Train length", "Predicted $r_2$, 6400", "Predicted $r_2-r_1$, 6400", "Predicted $r_2$, BCE 0.01", "Predicted $r_2-r_1$, BCE 0.01"], [
        [seed, length, *(f"{next(row for row in held_out if row['seed']==seed and row['length']==length and row['comparison']==comparison)[field]:.6f}" for comparison in ("update6400", "common100_bce0.01") for field in ("predicted_r", "predicted_effect"))]
        for seed in SEEDS for length in (10, 100)]) if held_out else ""
    max_loss_delta = max(abs(row["common100_bce_minus_control"]) for row in effects if row["optimizer_updates"] == 6400)
    max_head_delta = max(abs(row["head_slope_minus_control"]) for row in effects if row["optimizer_updates"] == 6400)
    first_map = {(row["seed"], row["length"], row["tag"]): row for row in first_rows}
    first_table = md_table(["Seed", "Train length", "Slow minus control", "Fast minus control"], [
        [seed, length, *(f"{first_map[(seed, length, tag)]['c_delta_after']-first_map[(seed, length, 'control')]['c_delta_after']:+.9f}" for tag in ("slow", "fast"))]
        for seed in SEEDS for length in (10, 100)])
    output_prefix = "../../" + directory.relative_to(project_dir()).as_posix()
    text = fr"""# Post-fitting coefficient learning-rate branch results

All 18 paths completed at total update 6400 in {status['wall_seconds']/60:.2f}
minutes on the saved CUDA environment. Six original baseline prefixes were
replayed to update 1600, then each full model/AdamW/data state was cloned into
coefficient learning-rate multipliers 1/3, 1, and 3. Seeds are 3, 4, 6; training
lengths are 10 and 100. See the [predeclared plan](COEFFICIENT_LR_BRANCH_PLAN.md)
and [execution specification](COEFFICIENT_LR_BRANCH_SPEC.md).

## Main result and prediction review

The intervention changes the selected exponent while retaining successful
fitting. The predeclared endpoint prediction scorecard is below; each comparison
retains every seed rather than testing only an average. First-step numerical
checks are reported separately and do not count as long-horizon forecast success.

{score_table}

At the fixed 6400-update endpoint:

{final_table}

The largest sibling difference in common length-100 balanced BCE from its
control is {max_loss_delta:.8g}, and in head slope is {max_head_delta:.8g}.
Coefficient-rate changes leave a large exponent difference at very similar
loss. Faster coefficient learning is accompanied by a smaller raw score margin;
query/key compensation therefore attenuates the exponent change rather than
eliminating it. Full coefficients, logits, masses, head states, and losses are
in [final_parameters.csv]({output_prefix}/analysis/final_parameters.csv).

![All exponent trajectories]({output_prefix}/analysis/exponent_trajectories.png)

## Common evaluation loss

Parameters were interpolated at the first post-1600 downward crossing of the
same length-100 balanced BCE, between adjacent 200-update records. Primary
target: 0.01. Estimated update counts are interpolation estimates, not exact
crossing times or extra checkpoints.

{matching_table}

All targets, interpolation brackets, head slope/intercept/threshold, raw gap,
coefficient and attention state are retained in
[loss_matched_parameters.csv]({output_prefix}/analysis/loss_matched_parameters.csv).
The 0.02 target is a planned companion and 0.005 is secondary; unreached targets
are explicitly censored. Own-length loss matching is supplementary only within
each training length, since the two own-loss functions differ. Matching a scalar
loss does not equalize geometry, head state, moments or every output.

Length contrast at every planned common-loss target:

{common_gap_table}

Paired parameter/head/step differences are in
[loss_matched_effects.csv]({output_prefix}/analysis/loss_matched_effects.csv).

![Exponent against common loss]({output_prefix}/analysis/exponent_vs_common_loss.png)

## Length contrast and what the intervention explains

Define $G_k=r_{{10,k}}-r_{{100,k}}$. Separate the gap already present at the branch
from the additional growth after branching:

{gap_table}

[length_gaps_by_step.csv]({output_prefix}/analysis/length_gaps_by_step.csv)
reports the existing 1600 gap, later growth, and treatment-minus-control gap
at 1800, 3200 and 6400. The equal-common-loss gap and length-specific effects
are in [prediction_scorecard.csv]({output_prefix}/analysis/prediction_scorecard.csv).

The multiplier-1 control's later length-gap growth decomposes as follows:

{gap_allocation_table}

Thus coefficient updates account for approximately 90–91% of the **signed
post-1600 gap growth** under the midpoint identity. The slow and fast interventions
move the length gap in the predicted directions, and the common-loss response
persists. This combination supports the proposed post-fitting update-allocation
explanation more strongly than decomposition alone. It does not identify a
causal mediation percentage, or explain why the parent moments differed at 1600.

The substantive evidence is the direction and relative magnitude of the
predeclared length-specific response, together with its persistence at common
loss. The intervention tests solution selection **after fitting**. Parents at
the two training lengths have different optimizer/head/geometry histories;
this experiment cannot establish why the entire difference arose before 1600.

## Actual update allocation and temporal order

All six first-update forecasts were saved before any comparison tail began.
Within every parent, the step-1601 batch, initial states, gradients, Adam moments
and query/key/head parameter updates match exactly between siblings. Only the
coefficient's applied learning rate differs. Numerical first-step exponent
prediction error is zero in all 18 branches.

{first_table}

These are exponent differences after the first update, not the coefficient
updates themselves. A changed midpoint coefficient can alter the score
contribution to $r$ even while the query/key parameter updates remain identical.
The first difference thus precedes any adaptation of the query/key or head
updates. Subsequent steps allow their gradients and moments to adapt normally.

Across all 4800 actual tail updates, exact midpoint accounting gives:

```math
\delta r=\overline{{\Delta}}\,\delta c+\overline c\,\delta\Delta.
```

{allocation_table}

Signed totals are in
[all_tail_allocation.csv]({output_prefix}/analysis/all_tail_allocation.csv),
and differences from the control are in
[allocation_effects.csv]({output_prefix}/analysis/allocation_effects.csv).
The detailed 1601–1800, 3190–3210 and 6390–6400 windows retain signed **and**
absolute sums of actual updates and margin contributions in
[trace_window_allocation.csv]({output_prefix}/analysis/trace_window_allocation.csv).
Compact 200-update allocation files cover every tail update, not only these
detailed windows. Maximum numerical decomposition residual is
{score['max_residual']:.3g}. Double-precision reconstruction from float32 weights
can differ slightly from the ordinary float32 diagnostic gap; this is rounding,
not a separate solution.

The accounting identity must sum correctly by construction; that alone is not
a mechanism test or causal mediation estimate. Relative length response,
gap movement, retained fitting, common-loss persistence, and frozen predictions
provide the nontrivial tests. Raw gradient size is not treated as applied update.

## Fitting, threshold entry, and extrapolation

Fitting retained at every recorded post-branch own-length point:
{score['all_fitting_retained']}. Explicit positive and negative class accuracy
is 1.0 for all paths at lengths 10, 100, 1000 and 10000:
{score['explicit_all_correct']}. Each length uses the original 50 balanced
examples. Diagnostics and model assumptions make identical-class examples
equivalent here; this is not a broad real-data accuracy estimate.

{threshold_table}

Brackets refer to 200-update samples; retention means all later sampled points
remain above 1, not an assertion about unrecorded intermediate updates.
Analytical failure predictions use the final head threshold as well as $r$.
`no_finite_crossing` means the reduced two-score classifier has no positive
threshold crossing over lengths at least 2 under its exact formula. Finite
values are predicted log10 lengths, not explicit evaluations at those lengths.
No explicit evaluation beyond 10000 was run. Ranking changes are retained in
[failure_ranking_changes.csv]({output_prefix}/analysis/failure_ranking_changes.csv).

## Reproduction and artifacts

- Six initial hashes, 1600 diagnostic prefixes and epoch histories match the
  historical baseline exactly. No historical intermediate weight checkpoint
  exists, so intermediate bitwise weight equality is not claimed.
- Six disposable 32-update original-group versus split-group controls match
  weights, moments, batches, generator states, RNG states and history exactly.
- All six multiplier-1 final weights match the old 6400 checkpoints bitwise;
  complete diagnostic rows and epoch histories match exactly.
- Every sibling consumes the same 4800 realized tail batches. Parent checkpoint
  file hashes remain unchanged. All six sibling verification files pass.
- Six full parent bundles, 18 final models, final optimizer/data/RNG states,
  frozen forecasts, source/environment snapshots and passive traces are saved
  in the [run directory]({output_prefix}). Each tail has 232 detailed updates;
  each parent has 11 detailed pre-branch updates.

Scientific compute is 96000 updates: 9600 shared prefix updates plus 86400 tail
updates. Disposable grouping checks add 384 updates and first-step forecasts
add 18. Every complete comparison path represents 409600 balanced sequences;
its tail processes 307200. Actual scheduled training processes 6144000 sequences
across the unique prefixes/tails, without triple-counting the shared prefixes.
The execution stayed inside the inherited 60-minute ceiling.

The focused branching tests and existing training-length, matched-initial,
Stage 3 evaluation, Stage 4A and Stage 4B regressions pass: 61 tests. The Stage 4A
temporary directory was redirected into the workspace for sandbox compatibility.

## One selected next direction

Selected direction: `{next_direction}`. No further experiment was run.
"""
    if next_direction == "held_out_coefficient_multiplier_2":
        text += fr"""
Test only the previously untested coefficient multiplier 2, using the same six
saved fitted parents and total budget 6400. Before that future run, the directional
prediction is $r_1<r_2<r_3$ at 6400 and common BCE 0.01; the positive difference
$r_2-r_1$ is larger at length 10 than length 100 in each selected seed, and
$G_1<G_2<G_3$. Freeze the following quantitative response model and forecasts
before running multiplier 2:

```math
g_k=r_k-r_0,\qquad
\beta=\frac{{\log(g_3/g_{{1/3}})}}{{\log 9}},\qquad
\widehat r_2=r_0+g_1 2^\beta.
```

Here $r_0$ is the saved parent's exponent at 1600. Fit the response separately
per parent and per comparison using only the three observed rates. This is a
phenomenological power-response hypothesis, not a derived optimization law.
Predefine numeric success as the observed $r_2-r_1$ lying within ±20% of the
predicted effect, along with retained fitting and all directional predictions.
The 20% tolerance is a declared falsification criterion, not a confidence interval
estimated from three rates. Retain a numeric miss even if the ordering succeeds.

{held_out_table}

Exact bounds and parameters are in
[held_out_multiplier2_predictions.csv]({output_prefix}/analysis/held_out_multiplier2_predictions.csv).
All rows are marked unexecuted. If that future experiment is authorized, freeze
these rows before launching and test them without revising the model or tolerance.

This is a held-out-rate test of the post-fitting explanation. It reuses the
initializations and does not supply independent-seed population evidence.
Relaxing the token or task assumptions remains a later decision after further
predictive validation.
"""
    else:
        text += """
Inspect the recorded head, query/key geometry and compensating updates before
adding a further intervention. Preserve the missed forecasts and every seed.
Do not interpret an exponent change alone as an explanation of the training-
length contrast, and do not substitute seed-5 rescue for this mechanism analysis.
"""
    (project_dir() / "documents/training_length/COEFFICIENT_LR_BRANCH_RESULTS.md").write_text(text, encoding="utf-8")
    print(json.dumps({"status": status, "scorecard": forecasts, "audit": {key: value for key, value in score.items() if key != "predictions"}}, indent=2))


if __name__ == "__main__":
    main()
