"""Score independent-seed predictions and inspect local optimizer amplification."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_coefficient_lr_branches import interpolate_loss, md_table
from analyze_coefficient_lr_holdout import load_trajectory
from run_coefficient_lr_holdout import file_digest
from run_training_length_pilot import sigmoid
from stage3_simplified_attention import project_dir, write_csv, write_json


def first_response(control_dir, x2_dir, seed, length):
    a = read_rows(control_dir / "update_trace.csv")[0]
    b = read_rows(x2_dir / "update_trace.csv")[0]
    u = float(a["u_before"])
    slope = sigmoid(u)
    adam_direction = float(a["u_normalized_direction"])
    delta_after = float(a["delta_after"])
    actual_response = float(b["c_delta_after"]) - float(a["c_delta_after"])
    local_response = delta_after * slope * 0.003 * adam_direction
    mass = float(a[f"target_mass_{length}_before"])
    logit = float(a[f"positive_logit_{length}_before"])
    rho = int(a["positives"]) / int(a["batch_size"])
    gradient_formula = rho * (sigmoid(logit) - 1) * float(a["head_slope_before"]) * mass * (1-mass) * float(a["delta_before"]) * math.log1p(length) * slope
    measured_gradient = float(a["u_gradient"])
    q = json.loads(a["q_before"])
    kd = json.loads(a["kd_before"])
    qnorm, knorm = math.sqrt(sum(x*x for x in q)), math.sqrt(sum(x*x for x in kd))
    cosine = sum(x*y for x, y in zip(q, kd)) / (qnorm * knorm)
    return {"seed": seed, "length": length, "u": u, "c": float(a["c_before"]),
        "softplus_derivative": slope, "delta_before": float(a["delta_before"]), "delta_after": delta_after,
        "query_norm": qnorm, "key_difference_norm": knorm, "query_key_cosine": cosine,
        "own_target_mass": mass, "own_positive_logit": logit, "positive_fraction": rho,
        "u_gradient": measured_gradient, "chain_gradient_prediction": gradient_formula,
        "chain_gradient_relative_error": gradient_formula / measured_gradient - 1 if measured_gradient else None,
        "adam_moment_after": float(a["u_moment_after"]), "adam_variance_after": float(a["u_variance_after"]),
        "adam_normalized_direction": adam_direction, "first_actual_u_update": float(a["u_update"]),
        "first_x2_minus_control_r": actual_response, "linearized_response": local_response,
        "linearized_relative_error": local_response / actual_response - 1 if actual_response else None}


def read_rows(path):
    import csv
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = project_dir()
    directory = root / args.run_dir
    status = json.loads((directory / "status.json").read_text())
    assert status["complete"] and status["completed_branches"] == 8 and not status["seeds_replaced"]
    freeze = json.loads((directory / "prediction_freeze.json").read_text())
    assert freeze["all_controls_completed"] and freeze["no_x2_tail_started"]
    assert file_digest(directory / "rule_frozen_before_new_training.json") == freeze["rule_sha256"]
    assert file_digest(directory / "predictions_frozen_before_x2.csv") == freeze["numeric_predictions_sha256"]
    rule = json.loads((directory / "rule_frozen_before_new_training.json").read_text())
    for source, digest in rule["source_sha256"].items():
        assert file_digest(Path(source)) == digest
    analysis = directory / "analysis"
    analysis.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), directory / "source_snapshot" / Path(__file__).name)
    eligibility = read_rows(directory / "parent_eligibility.csv")
    gate = {(int(row["seed"]), int(row["length"])): row["eligible"] == "True" for row in eligibility}
    paths, endpoints, allocation, matches = {}, [], [], []
    verification = []
    for seed in (8, 9):
        for length in (10, 100):
            parent = f"n{length}_s{seed}"
            e = json.loads((directory / "parents" / parent / "eligibility.json").read_text())
            assert file_digest(directory / "parents" / parent / "parent_state.pt") == e["checkpoint_file_sha256"]
            group_check = json.loads((directory / "validation" / parent / "group_verification.json").read_text())
            assert all(value is True for key, value in group_check.items() if key != "steps")
            sibling_check = json.loads((directory / "branches" / f"{parent}_x2/sibling_verification.json").read_text())
            assert all(sibling_check.values())
            for tag in ("control", "x2"):
                run = directory / "branches" / f"{parent}_{tag}"
                check = json.loads((run / "branch_verification.json").read_text())
                assert check["parent_sha256"] == e["parent_sha256"] and check["tail_updates"] == 4800 and check["traced_updates"] == 232
                assert check["max_decomposition_residual"] < 1e-10
                verification.append({"seed": seed, "length": length, "tag": tag,
                    "all_checks_passed": True, "max_residual": check["max_decomposition_residual"]})
                path = load_trajectory(run, length)
                paths[(seed, length, tag)] = path
                explicit = read_rows(run / "metrics_by_length.csv")
                entry = next((i for i, row in enumerate(path) if row["c_delta"] > 1), None)
                endpoints.append({"seed": seed, "length": length, "tag": tag, "eligible_parent": gate[(seed, length)],
                    **path[-1], "fitting_retained_on_grid": all(row["fitting_correct"] for row in path),
                    "explicit_all_classes_correct": all(float(row["positive_accuracy"]) == float(row["negative_accuracy"]) == 1 for row in explicit),
                    "first_above": path[entry]["optimizer_updates"] if entry is not None else None,
                    "previous_recording": path[entry-1]["optimizer_updates"] if entry is not None else None,
                    "later_sampled_points_above": all(row["c_delta"] > 1 for row in path[entry:]) if entry is not None else False})
                allocation.append({"seed": seed, "length": length, "tag": tag, **check["total_allocation"]})
                for target in (0.02, 0.01, 0.005):
                    result = interpolate_loss(path, "common100_bce", target)
                    matches.append({"seed": seed, "length": length, "tag": tag, "target_bce": target,
                        "reached": result is not None, "values": result})
    write_csv(analysis / "final_parameters.csv", endpoints)
    write_csv(analysis / "allocation.csv", allocation)
    write_json(analysis / "common_loss_parameters.json", {"matches": matches})
    write_csv(analysis / "verification.csv", verification)
    write_csv(analysis / "trajectories.csv", [{"seed": seed, "length": length, "tag": tag, **row}
        for (seed, length, tag), path in paths.items() for row in path])
    numerical = []
    for prediction in read_rows(directory / "predictions_frozen_before_x2.csv"):
        seed, length = int(prediction["seed"]), int(prediction["length"])
        comparison = prediction["comparison"]
        control = paths[(seed, length, "control")]
        changed = paths[(seed, length, "x2")]
        a = control[-1] if comparison == "update6400" else interpolate_loss(control, "common100_bce", 0.01)
        b = changed[-1] if comparison == "update6400" else interpolate_loss(changed, "common100_bce", 0.01)
        available = prediction["evaluable"] == "True" and a is not None and b is not None
        if a is not None:
            assert a["c_delta"] == float(prediction["control_r"])
        effect = b["c_delta"] - a["c_delta"] if available else None
        expected = float(prediction["predicted_effect"]) if prediction["predicted_effect"] else None
        numerical.append({"seed": seed, "length": length, "comparison": comparison, "evaluable": available,
            "reason": prediction["reason"] if b is not None else "x2_common_target_censored",
            "predicted_r": float(prediction["predicted_r"]) if prediction["predicted_r"] else None,
            "actual_r": b["c_delta"] if b else None, "control_r": a["c_delta"] if a else None,
            "predicted_effect": expected, "actual_effect": effect,
            "effect_lower": float(prediction["effect_lower"]) if prediction["effect_lower"] else None,
            "effect_upper": float(prediction["effect_upper"]) if prediction["effect_upper"] else None,
            "relative_effect_error": effect / expected - 1 if available else None,
            "numeric_passed": float(prediction["effect_lower"]) <= effect <= float(prediction["effect_upper"]) if available else None,
            "positive_effect": effect > 0 if available else None})
    directions, length_gaps = [], []
    for seed in (8, 9):
        for comparison in ("update6400", "common100_bce0.01"):
            members = [next(row for row in numerical if row["seed"] == seed and row["length"] == length and row["comparison"] == comparison) for length in (10, 100)]
            available = all(row["evaluable"] for row in members)
            directions.append({"seed": seed, "comparison": comparison, "evaluable": available,
                "effect10": members[0]["actual_effect"], "effect100": members[1]["actual_effect"],
                "larger_effect_at10": members[0]["actual_effect"] > members[1]["actual_effect"] > 0 if available else None})
            length_gaps.append({"seed": seed, "comparison": comparison,
                "control_gap": members[0]["control_r"]-members[1]["control_r"] if available else None,
                "x2_gap": members[0]["actual_r"]-members[1]["actual_r"] if available else None,
                "predicted_effect_gap": members[0]["predicted_effect"]-members[1]["predicted_effect"] if available else None,
                "actual_effect_gap": members[0]["actual_effect"]-members[1]["actual_effect"] if available else None})
    write_csv(analysis / "numeric_predictions.csv", numerical)
    write_csv(analysis / "length_effect_predictions.csv", directions)
    write_csv(analysis / "length_gaps.csv", length_gaps)

    # Local first-step response separates raw margin, softplus derivative, and Adam scaling.
    original = root / "runs/training_length/coefficient_lr_branches_20261010_01"
    holdout = root / "runs/training_length/coefficient_lr_multiplier2_20261010_02"
    local = []
    for seed in (3, 4, 6, 8, 9):
        for length in (10, 100):
            control_dir = (original if seed < 8 else directory) / "branches" / f"n{length}_s{seed}_control"
            changed_dir = (holdout if seed < 8 else directory) / "branches" / f"n{length}_s{seed}_x2"
            local.append(first_response(control_dir, changed_dir, seed, length))
    local_ratios = []
    for seed in (3, 4, 6, 8, 9):
        a, b = [next(row for row in local if row["seed"] == seed and row["length"] == length) for length in (10, 100)]
        local_ratios.append({"seed": seed, "delta_ratio_10_over100": a["delta_after"] / b["delta_after"],
            "softplus_derivative_ratio_10_over100": a["softplus_derivative"] / b["softplus_derivative"],
            "adam_direction_ratio_10_over100": a["adam_normalized_direction"] / b["adam_normalized_direction"],
            "raw_u_gradient_ratio_10_over100": abs(a["u_gradient"] / b["u_gradient"]),
            "second_moment_ratio_100_over10": b["adam_variance_after"] / a["adam_variance_after"],
            "first_moment_magnitude_ratio_10_over100": abs(a["adam_moment_after"] / b["adam_moment_after"]),
            "response_ratio_if_Adam_denominators_equal": (a["delta_after"] / b["delta_after"])
                * (a["softplus_derivative"] / b["softplus_derivative"])
                * abs(a["adam_moment_after"] / b["adam_moment_after"]),
            "linearized_response_ratio_10_over100": a["linearized_response"] / b["linearized_response"],
            "actual_response_ratio_10_over100": a["first_x2_minus_control_r"] / b["first_x2_minus_control_r"]})
    write_csv(analysis / "local_optimizer_states.csv", local)
    write_csv(analysis / "first_response_factorization.csv", local_ratios)
    early_memory = []
    for seed in (3, 4):
        for length in (10, 100):
            early_file = root / "runs/training_length/update_trace_20260929_01" / f"learned_log_n{length}_s{seed}_baseline/update_trace.csv"
            early = [row for row in read_rows(early_file) if int(row["optimizer_updates"]) <= 200]
            assert {int(row["optimizer_updates"]) for row in early} == set(range(1, 201))
            row = read_rows(original / "branches" / f"n{length}_s{seed}_control/update_trace.csv")[0]
            v_at1600 = float(row["u_variance_before"])
            contribution = sum(0.001 * 0.999 ** (1600 - int(item["optimizer_updates"])) * float(item["u_gradient"]) ** 2 for item in early)
            early_memory.append({"seed": seed, "length": length, "v_at1600": v_at1600,
                "weighted_first200_contribution": contribution, "fraction_of_current_variance": contribution / v_at1600,
                "source": str(early_file), "source_sha256": file_digest(early_file)})
    write_csv(analysis / "early_gradient_memory.csv", early_memory)

    primary = [row for row in numerical if row["comparison"] == "update6400"]
    secondary = [row for row in numerical if row["comparison"] == "common100_bce0.01"]
    audit = {"complete": True, "eligible_parents": sum(gate.values()), "planned_parents": 4,
        "primary_passes": sum(row["numeric_passed"] is True for row in primary),
        "primary_evaluable": sum(row["evaluable"] for row in primary),
        "secondary_passes": sum(row["numeric_passed"] is True for row in secondary),
        "secondary_evaluable": sum(row["evaluable"] for row in secondary),
        "positive_effect_passes": sum(row["positive_effect"] is True for row in numerical),
        "length_effect_passes": sum(row["larger_effect_at10"] is True for row in directions),
        "all_fitting_retained": all(row["fitting_retained_on_grid"] for row in endpoints),
        "all_explicit_classes_correct": all(row["explicit_all_classes_correct"] for row in endpoints),
        "max_absolute_relative_effect_error": max((abs(row["relative_effect_error"]) for row in numerical if row["evaluable"]), default=None),
        "rule_and_numeric_forecasts_unchanged": True, "no_new_seed_refit": True,
        "max_local_linearization_relative_error": max(abs(row["linearized_relative_error"]) for row in local),
        "max_chain_gradient_relative_error": max(abs(row["chain_gradient_relative_error"]) for row in local),
        "wall_seconds": status["wall_seconds"], "next_direction": "explain_length_response_with_geometry_and_optimizer_history"}
    audit["overall_confirmation"] = (audit["eligible_parents"] == audit["primary_passes"] == audit["secondary_passes"] == 4
        and audit["positive_effect_passes"] == 8 and audit["length_effect_passes"] == 4 and audit["all_fitting_retained"])
    write_json(analysis / "audit.json", audit)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for comparison, axis, title in zip(("update6400", "common100_bce0.01"), axes, ("At update 6400 (primary)", "At common BCE 0.01 (secondary)")):
        group = [row for row in numerical if row["comparison"] == comparison]
        axis.axhspan(-20, 20, color="#009E73", alpha=0.12)
        axis.axhline(0, color="gray", linewidth=0.8)
        for i, row in enumerate(group):
            if row["evaluable"]:
                axis.scatter(i, 100 * row["relative_effect_error"], color="#0072B2" if row["length"] == 10 else "#D55E00", s=60)
        axis.set_xticks(range(4), [f"s{row['seed']} / L{row['length']}" for row in group])
        axis.set(title=title, ylabel="Error in predicted effect (%)")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Independent seeds: frozen old-seed normalized-effect rule, ±20% criterion")
    fig.tight_layout()
    fig.savefig(analysis / "independent_prediction_errors.png", dpi=170)
    plt.close(fig)
    prefix = "../../" + directory.relative_to(root).as_posix()
    tables = {}
    for comparison in ("update6400", "common100_bce0.01"):
        tables[comparison] = md_table(["Seed", "Train length", "Predicted $r_2$", "Observed $r_2$", "Predicted effect", "Observed effect", "Effect error", "Passed"], [
            [row["seed"], row["length"], f"{row['predicted_r']:.6f}" if row["evaluable"] else "Inapplicable",
             f"{row['actual_r']:.6f}" if row["actual_r"] is not None else "Censored",
             f"{row['predicted_effect']:.6f}" if row["evaluable"] else "Inapplicable",
             f"{row['actual_effect']:.6f}" if row["evaluable"] else "Inapplicable",
             f"{100*row['relative_effect_error']:+.2f}%" if row["evaluable"] else "Inapplicable", row["numeric_passed"]]
            for row in numerical if row["comparison"] == comparison])
    gate_table = md_table(["Seed", "Train length", "Fitted at 1600", "Positive logit", "Negative logit", "$r$ at branch"], [
        [row["seed"], row["length"], row["eligible"], f"{float(row['positive_logit']):+.4f}", f"{float(row['negative_logit']):+.4f}", f"{float(row['c_delta']):.6f}"] for row in eligibility])
    direction_table = md_table(["Seed", "Comparison", "Effect at 10", "Effect at 100", "Larger at 10"], [
        [row["seed"], row["comparison"], f"{row['effect10']:.6f}" if row["evaluable"] else "Inapplicable",
         f"{row['effect100']:.6f}" if row["evaluable"] else "Inapplicable", row["larger_effect_at10"]] for row in directions])
    final_table = md_table(["Seed", "Train length", "Rate", "$r$", "First $r>1$ bracket", "Later grid above", "Analytical failure"], [
        [row["seed"], row["length"], 1 if row["tag"] == "control" else 2, f"{row['c_delta']:.6f}",
         f"({row['previous_recording']}, {row['first_above']}]" if row["first_above"] else "Not observed", row["later_sampled_points_above"],
         f"log10 length {row['failure_log10']:.3f}" if row["failure_status"] == "finite" else row["failure_status"]] for row in endpoints])
    local_table = md_table(["Seed", r"$\Delta^+_{10}/\Delta^+_{100}$", "Softplus derivative ratio", "Adam direction ratio", "Raw gradient magnitude ratio", "Variance 100 / 10", "First response 10 / 100"], [
        [row["seed"], *(f"{row[key]:.4f}" for key in ("delta_ratio_10_over100", "softplus_derivative_ratio_10_over100",
            "adam_direction_ratio_10_over100", "raw_u_gradient_ratio_10_over100", "second_moment_ratio_100_over10", "actual_response_ratio_10_over100"))] for row in local_ratios])
    memory_table = md_table(["Seed", "Train length", "Fraction of step-1600 variance from updates 1–200"], [
        [row["seed"], row["length"], f"{100*row['fraction_of_current_variance']:.6g}%"] for row in early_memory])
    denominator_table = md_table(["Seed", "First-moment magnitude 10 / 100", "Local response ratio if denominators equal"], [
        [row["seed"], f"{row['first_moment_magnitude_ratio_10_over100']:.4f}", f"{row['response_ratio_if_Adam_denominators_equal']:.4f}"] for row in local_ratios])
    effects_allocation = []
    for seed in (8, 9):
        for length in (10, 100):
            a = next(row for row in allocation if row["seed"] == seed and row["length"] == length and row["tag"] == "x2")
            b = next(row for row in allocation if row["seed"] == seed and row["length"] == length and row["tag"] == "control")
            effects_allocation.append({"seed": seed, "length": length,
                **{key + "_minus_control": a[key] - b[key] for key in ("c_part", "score_part", "r_change")}})
    write_csv(analysis / "allocation_effects.csv", effects_allocation)
    allocation_effect_table = md_table(["Seed", "Train length", "Coefficient contribution difference", "Score contribution difference", "Total exponent effect"], [
        [row["seed"], row["length"], *(f"{row[key]:+.6f}" for key in ("c_part_minus_control", "score_part_minus_control", "r_change_minus_control"))] for row in effects_allocation])
    growth_factorization = []
    for seed in (8, 9):
        for comparison in ("update6400", "common100_bce0.01"):
            members = [next(row for row in numerical if row["seed"] == seed and row["length"] == length and row["comparison"] == comparison) for length in (10, 100)]
            if all(row["evaluable"] for row in members):
                growth = [row["control_r"] - paths[(seed, length, "control")][0]["c_delta"] for row, length in zip(members, (10, 100))]
                normalized = [row["actual_effect"] / value for row, value in zip(members, growth)]
                growth_factorization.append({"seed": seed, "comparison": comparison,
                    "control_growth_ratio_10_over100": growth[0]/growth[1],
                    "normalized_response_ratio_10_over100": normalized[0]/normalized[1],
                    "actual_effect_ratio_10_over100": members[0]["actual_effect"]/members[1]["actual_effect"]})
    write_csv(analysis / "endpoint_response_factorization.csv", growth_factorization)
    growth_table = md_table(["Seed", "Comparison", "Control-growth ratio", "Normalized-response ratio", "Actual effect ratio"], [
        [row["seed"], row["comparison"], *(f"{row[key]:.4f}" for key in ("control_growth_ratio_10_over100", "normalized_response_ratio_10_over100", "actual_effect_ratio_10_over100"))] for row in growth_factorization])
    report = fr"""# Independent-seed coefficient-rate results

Eight paths for seeds 8 and 9 completed at total update 6400 in
{status['wall_seconds']/60:.2f} minutes. Four shared prefixes were trained through
1600, then full model/AdamW/data/RNG states were cloned into coefficient-rate
multipliers 1 and 2. Training lengths remained 10 and 100. Query/key and head
learning rates stayed 0.003. See the [launch specification](COEFFICIENT_LR_NEWSEEDS_SPEC.md).
No new-seed 1/3 or 3 multiplier was run, and no seed was replaced.

## Parent eligibility

{gate_table}

Eligibility includes both own-length classes strictly correct, positive head
slope and positive raw margin. Every intended parent is retained in the table.
Failed eligibility would make its pair descriptive rather than a confirmation
of a post-fitting prediction; it would count against full four-parent success.

## Frozen rule and numerical predictions

The existing seed-3/4/6 effect divided by its control's post-branch exponent
growth was averaged separately by length and comparison. This normalized-effect
rule is distinct from the previous power-response fit. Its four coefficients
were hashed before any new-seed training: 0.7521685749951815 / 0.6264930352686643
at 6400, and 0.8575632252231385 / 0.6842067538150549 at common BCE 0.01, for
training lengths 10 / 100 respectively. No coefficient was refitted per new seed.

All four new control paths completed before any multiplier-2 tail. Their
post-branch growth was the only new outcome used to construct the stored effect
forecast. All eight forecasts and ±20% effect bounds were saved before any
multiplier-2 training. The exact source hashes and calibration data remain unchanged.
This is a conditional prediction given the new control path, not a prediction
of that control's optimization trajectory from initialization.

Primary endpoint, update 6400: {audit['primary_passes']}/{audit['primary_evaluable']}
evaluable forecasts passed, out of four planned pairs.

{tables['update6400']}

Secondary endpoint, common length-100 balanced BCE 0.01:
{audit['secondary_passes']}/{audit['secondary_evaluable']} evaluable forecasts passed,
out of four planned pairs.

{tables['common100_bce0.01']}

Common-loss parameters use the first post-branch downward crossing interpolated
between adjacent 200-update records. The same predeclared ±20% criterion applies
to the effect $r_2-r_1$, not the absolute exponent. Censored or inapplicable
forecasts remain in [numeric_predictions.csv]({prefix}/analysis/numeric_predictions.csv)
with reasons. No tolerance, target or seed was adjusted after the outcome.

![Independent-seed prediction errors]({prefix}/analysis/independent_prediction_errors.png)

## Direction, fitting, and extrapolation

{direction_table}

Positive exponent effect passed {audit['positive_effect_passes']}/8 planned
comparisons. Larger effect at length 10 passed {audit['length_effect_passes']}/4.
Own-length fitting retained at every recorded post-branch point:
{audit['all_fitting_retained']}. All explicit positive/negative accuracies at
10, 100, 1000, 10000 are 1.0: {audit['all_explicit_classes_correct']}.

{final_table}

Threshold-entry brackets and retention refer to the 200-update grid. Larger
lengths and failure log10 lengths are analytical predictions under the reduced
two-score classifier, including head threshold; no explicit evaluation beyond
10000 was performed. Full parameters, head state, common/own losses, exposure,
232 detailed updates per tail and compact signed allocation across all 4800
tail updates are retained. This confirms only the specified seeds and model.

## Why length 10 responds more strongly: local optimizer analysis

Analyze recorded states without another intervention. At update 1601, sibling
weights, batch, gradients and Adam moments match; query/key/head parameter
updates match. Only the coefficient learning rate differs. For the first
coefficient-rate intervention, a first-order factorization is

```math
A_L(2,1601)\approx\Delta_L^+\,\sigma(u_L)\,\eta_u D_L,
\qquad D_L=-\frac{{\widehat m_L}}{{\sqrt{{\widehat v_L}}+\epsilon}}.
```

Here $\sigma(u)$ is the softplus derivative, $\eta_u=0.003$ is the baseline
coefficient learning rate, and $D$ is the actual Adam normalized direction.
The finite first-step difference is also recorded directly, using float32
optimizer updates rather than assuming a perfect multiplier-to-step equality.
Maximum relative error of the linearized response across old and new parents:
{100*audit['max_local_linearization_relative_error']:.3f}%.

{local_table}

Across all five seeds, length 10 has a **smaller** raw margin and a **smaller**
instantaneous coefficient-gradient magnitude. Those quantities alone therefore
do not explain its larger first-step response. Its softplus derivative is
1.29–1.36 times larger, and its applied Adam normalized direction is 2.02–2.14
times larger. The length-100 coefficient second moment is 7.63–9.20 times larger;
its larger normalization denominator outweighs its larger current gradient and
first moment. Together the three factors reproduce the 2.22–2.39 response ratio.

As a conditional arithmetic check, retain the measured first moments, raw
margins and softplus derivatives but make the two Adam denominators equal:

{denominator_table}

Each such local ratio is below 1. This is **not** a trained moment-matching
intervention or a causal estimate. It shows that the observed denominator
asymmetry is necessary to the greater length-10 response in this particular
first-step factorization; it does not prove that holding moments equal would
preserve fitting or the later optimization path.

The table compares length 10 with length 100 for each seed. It separates three
factors that are otherwise hidden in a raw-gradient comparison: raw margin,
softplus sensitivity, and Adam normalization. Geometry enters through the raw
margin; query/key norms and their cosine are also retained in
[local_optimizer_states.csv]({prefix}/analysis/local_optimizer_states.csv).
The factorization is a local conditional explanation at a fitted checkpoint,
not an identification of how all parent differences arose during the first 1600
updates. Subsequent query/key/head adaptation can attenuate or amplify the effect;
all-update control-minus-treatment allocation is available for that distinction.

For this two-token task, the negative class contributes no analytical coefficient
gradient. With positive batch fraction $\rho$, positive attention mass $p$,
positive logit $z$ and head slope $h$, the coefficient gradient follows

```math
g_u=\rho[\sigma(z)-1]h\,p(1-p)\Delta\log(1+L)\sigma(u).
```

Its maximum relative discrepancy from the recorded first-step gradient is
{100*audit['max_chain_gradient_relative_error']:.4f}%, consistent with numerical
rounding. This distinguishes attention/head/length effects on the raw gradient
from the optimizer's applied update. The current gradient and moment are not
interchangeable; the second moment remembers earlier squared gradients.

Existing old-seed traces cover every update 1–200 for seeds 3 and 4. Under the
unchanged Adam settings, their surviving contribution to variance at 1600 is

```math
v_{{1600,1:200}}=(1-\beta_2)\sum_{{j=1}}^{{200}}
\beta_2^{{1600-j}}g_j^2,\qquad\beta_2=0.999.
```

{memory_table}

The rest of the 1600-update variance is not reconstructed from sparse traces.
The first 200 updates account for at most 0.0031% of the step-1600 second moment
in these four old-parent cells. Thus the very earliest 1–200 gradients do **not**
explain the large variance difference. The relevant squared-gradient history
was accumulated predominantly during updates 201–1600, which are not fully
covered by the existing detailed traces. This rules out an earliest-phase
memory explanation and identifies the unresolved fitting-transition interval.
This early-history analysis covers only old seeds 3 and 4; it is not imputed
for seed 6 or new seeds 8/9. Source hashes and contributions are retained in
[early_gradient_memory.csv]({prefix}/analysis/early_gradient_memory.csv).

The complete post-branch exponent effect also shows compensating score updates:

{allocation_effect_table}

The negative score contribution attenuates the positive coefficient contribution
in all new pairs. This exact signed decomposition is an accounting identity,
not a causal mediation percentage. Head changes are recorded alongside it;
the first head update is identical across siblings, while later head adaptation
remains allowed. Endpoint head/geometry states are not assumed to match merely
because the common loss matches.

For the successful new pairs, factor the endpoint effect ratio into control
growth and normalized response, using the same denominator as the frozen rule:

{growth_table}

All ratios are length 10 divided by length 100. Much of the larger endpoint
effect comes from a larger baseline post-branch exponent increase; a further
length-specific normalized-response difference contributes. This is a
descriptive factorization, not an explanation of how the baseline growth or
moment history originated.

## Validity, compute, and next direction

All four full parent files remain unchanged. Same-seed initialization hashes
match across lengths. Four disposable 32-update original/split-group comparisons
match weights, moments, history, RNG, generators and batches exactly. All four
new sibling pairs have identical initial gradients/moments, first query/key/head
updates, full future batch hashes and prefix histories. Frozen rule and numerical
forecast hashes pass. Five serialized-continuation/isolation tests passed again.
Three additional tests verify the frozen growth-based forecast, failed-parent
handling and censoring without extrapolation.
No shared training code was modified. The process uses the original deterministic
kernel settings explicitly.

Scientific compute is 44800 updates and 2867200 sequence exposures, counting
four shared prefixes once and each tail. Sibling tails deliberately reuse batches.
Disposable grouping checks add 256 updates. Each
complete path represents 409600 balanced sequence exposures. Execution remains inside
the inherited 60-minute ceiling.

Overall four-parent confirmation, including eligibility, both numeric comparisons,
positive effects, larger length-10 response and retained fitting:
{audit['overall_confirmation']}. This independent-initialization result tests
transfer of a length-specific rule without new-model rate interpolation. It
does not establish a universal optimization law from two new seeds.

The next direction is explanation of the length response using geometry and
optimizer history, rather than expanding learning-rate choices. The read-only
analysis above starts that work by isolating immediate softplus/Adam amplification
and checking early-gradient memory on the available old traces. Future causal
controls would need to separate these factors without silently changing initial
behavior or optimizer history. No additional training or model-assumption change
was run after the eight paths.
"""
    (root / "documents/training_length/COEFFICIENT_LR_NEWSEEDS_RESULTS.md").write_text(report, encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
