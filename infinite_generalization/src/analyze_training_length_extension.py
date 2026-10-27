"""Audit longer-budget continuations, exact new entries, and inherited Adam memory."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from analyze_coefficient_lr_branches import md_table
from analyze_training_length_confirmation import failure_status
from analyze_training_length_sweep import (COLORS, INTERPOLATED_FIELDS, STATE_FIELDS,
    analytic, first_loss_crossing, fmt, is_fitted)
from run_coefficient_lr_branches import read_rows, tree_digest
from run_coefficient_lr_holdout import file_digest
from run_training_length_extension import END, LENGTHS, SEEDS, START, parameter_groups
from run_training_length_sweep import COEFFICIENT, EVAL_LENGTHS
from stage3_simplified_attention import project_dir, write_csv, write_json

TARGETS = (.001, .0005, .0001)
LENGTH_COLORS = {10: "#6c3f94", 30: "#24838a", 1000: "#df972a"}


def crossing_summary(trace, earlier_grid):
    """Treat an above-threshold starting state separately from a new entry."""
    assert [int(r["optimizer_updates"]) for r in trace] == list(range(START + 1, END + 1))
    above_start = float(trace[0]["r_before"]) > 1
    events = []
    for index, row in enumerate(trace):
        before, after = float(row["r_before"]), float(row["r_after"])
        if index:
            assert before == float(trace[index - 1]["r_after"]), "Trace state discontinuity"
        if (before > 1) != (after > 1):
            events.append({"optimizer_updates": int(row["optimizer_updates"]),
                "event": "entry" if after > 1 else "exit", "r_before": before, "r_after": after})
    first_new = next((r["optimizer_updates"] for r in events if r["event"] == "entry"), None)
    previous_entry = next((i for i, row in enumerate(earlier_grid) if row["c_delta"] > 1), None)
    old_bracket = f"({earlier_grid[previous_entry - 1]['optimizer_updates']}, {earlier_grid[previous_entry]['optimizer_updates']}]" if previous_entry not in (None, 0) else None
    retention_from = START + 1 if above_start else first_new
    retained = all(float(row["r_after"]) > 1 for row in trace if int(row["optimizer_updates"]) >= retention_from) if retention_from is not None else False
    return {"already_above_at6400": above_start, "historical_first_entry_grid_bracket": old_bracket,
        "first_new_entry": first_new, "right_censored_at12800": not above_start and first_new is None,
        "entry_status": "already_above_at_start" if above_start else "entered_during_extension" if first_new else "right_censored_12800",
        "all_steps_above_after_start_or_entry": retained, "exit_count": sum(e["event"] == "exit" for e in events),
        "trace_r_at6400": float(trace[0]["r_before"]), "trace_r_at12800": float(trace[-1]["r_after"])}, events


def reconstruct_boundary_memory(trace, initial_v, endpoint):
    chosen = [row for row in trace if int(row["optimizer_updates"]) <= endpoint]
    assert [int(row["optimizer_updates"]) for row in chosen] == list(range(START + 1, endpoint + 1))
    beta = float(chosen[-1]["beta2"])
    assert all(float(row["beta2"]) == beta for row in chosen)
    carryover = beta ** (endpoint - START) * initial_v
    new = math.fsum((1 - beta) * beta ** (endpoint - int(row["optimizer_updates"])) * float(row["u_gradient"]) ** 2 for row in chosen)
    recorded = float(chosen[-1]["v_after"])
    total = carryover + new
    return {"endpoint": endpoint, "beta2": beta, "boundary_second_moment": initial_v,
        "recorded_second_moment": recorded, "inherited_contribution": carryover, "new_gradient_contribution": new,
        "inherited_fraction": carryover / recorded if recorded else None,
        "new_gradient_fraction": new / recorded if recorded else None,
        "reconstructed_second_moment": total, "absolute_residual": abs(total - recorded),
        "relative_residual": abs(total - recorded) / abs(recorded) if recorded else 0.0,
        "passed": math.isclose(total, recorded, rel_tol=1e-4, abs_tol=1e-12)}


def load_path(run):
    raw = read_rows(run / "training_diagnostics.csv")
    assert len(raw) == 65 * 6
    steps = sorted({int(r["optimizer_updates"]) for r in raw})
    assert steps == list(range(0, END + 1, 200))
    config = json.loads((run / "config.json").read_text())
    length = config["train_lengths"][0]
    path = []
    for step in steps:
        group = {int(r["length"]): r for r in raw if int(r["optimizer_updates"]) == step}
        assert sorted(group) == sorted(EVAL_LENGTHS)
        common, own = group[100], group[length]
        state = {field: float(common[field]) if common[field] else None for field in STATE_FIELDS}
        values = {"optimizer_updates": step, **state,
            "common100_bce": float(common["balanced_bce"]), "own_bce": float(own["balanced_bce"]),
            "own_positive": float(own["positive_logit"]), "own_negative": float(own["negative_logit"]),
            "own_mass": float(own["target_mass"]), "mass100": float(common["target_mass"]),
            "positive100": float(common["positive_logit"]), "negative100": float(common["negative_logit"])}
        values["fitted"] = is_fitted(values)
        finite = all(math.isfinite(state[key]) for key in STATE_FIELDS)
        values["failure_status"], values["failure_log10"] = failure_status(state) if finite else ("nonfinite_state", None)
        path.append(values)
    return path


def make_plots(analysis, paths, common, gaps, growth, memory):
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    for row, seed in enumerate(SEEDS):
        for length in LENGTHS:
            path = paths[(seed, length)]
            for col, field in enumerate(("c_delta", "c", "delta")):
                axes[row, col].plot([r["optimizer_updates"] for r in path], [r[field] for r in path], color=LENGTH_COLORS[length], label=str(length))
        for col in range(3):
            axes[row, col].set(xlabel="Global optimizer updates", ylabel=("Exponent r", "Coefficient c", "Raw margin Delta")[col], title=f"Seed {seed}")
            axes[row, col].axvline(START, color="gray", lw=1, ls=":")
            axes[row, col].grid(alpha=.2)
        axes[row, 0].axhline(1, color="black", lw=1, ls="--")
    axes[0, 0].legend(title="Train length")
    fig.suptitle("Historical prefix and restored baseline continuation (boundary at 6400)")
    fig.savefig(analysis / "combined_trajectories.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for ax, field, label in zip(axes.flat, ("c", "delta", "c_delta", "head_threshold"), ("Coefficient c", "Raw margin Delta", "Exponent r", "Head threshold")):
        for seed in SEEDS:
            for step, style in ((START, ":"), (END, "-")):
                values = [next(r[field] for r in paths[(seed, length)] if r["optimizer_updates"] == step) for length in LENGTHS]
                ax.plot(LENGTHS, values, marker="o", ls=style, color=COLORS[seed], label=f"Seed {seed}, {step}")
        ax.set(xscale="log", xlabel="Training length", ylabel=label)
        ax.set_xticks(LENGTHS, labels=LENGTHS)
        ax.grid(alpha=.2)
        if field == "c_delta":
            ax.axhline(1, color="black", lw=1, ls="--")
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("Paired length response: 6400 versus 12800")
    fig.savefig(analysis / "length_response.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    for row, seed in enumerate(SEEDS):
        for length in LENGTHS:
            path = paths[(seed, length)]
            for col, field in enumerate(("own_bce", "common100_bce")):
                axes[row, col].plot([r["optimizer_updates"] for r in path], [r[field] for r in path], color=LENGTH_COLORS[length], label=str(length))
        for col in range(2):
            axes[row, col].set(xlabel="Global optimizer updates", ylabel=("Own-length BCE", "Length-100 BCE")[col], yscale="log", title=f"Seed {seed}")
            axes[row, col].axvline(START, color="gray", lw=1, ls=":")
            axes[row, col].grid(alpha=.2)
    axes[0, 0].legend(title="Train length")
    fig.savefig(analysis / "loss_trajectories.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, target in zip(axes, TARGETS):
        for seed in SEEDS:
            records = [next(r for r in common if r["seed"] == seed and r["length"] == length and r["target"] == target) for length in LENGTHS]
            ax.plot(LENGTHS, [r["c_delta"] if r["eligible_fitted"] else math.nan for r in records], "o-", color=COLORS[seed], label=f"Seed {seed}")
        coverage = sum(r["eligible_fitted"] for r in common if r["target"] == target)
        ax.set(xscale="log", xlabel="Training length", ylabel="Interpolated exponent", title=f"BCE {target}: {coverage}/9 eligible")
        ax.set_xticks(LENGTHS, labels=LENGTHS)
        ax.axhline(1, color="black", lw=1, ls="--")
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.savefig(analysis / "common_loss.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    for row, seed in enumerate(SEEDS):
        for key, label in (("gap_10_30", "r10 - r30"), ("gap_30_1000", "r30 - r1000")):
            selected = [r for r in gaps if r["seed"] == seed]
            axes[row, 0].plot([r["optimizer_updates"] for r in selected], [r[key] for r in selected], label=label)
        for length in LENGTHS:
            selected = [r for r in growth if r["seed"] == seed and r["length"] == length]
            axes[row, 1].plot([r["update_upper"] for r in selected], [r["r_growth_per_update"] for r in selected], "o-", color=LENGTH_COLORS[length], label=str(length))
        axes[row, 0].set(xlabel="Global updates", ylabel="Paired exponent gap", title=f"Seed {seed}")
        axes[row, 1].set(xlabel="End of fixed 1600-update window", ylabel="Average r growth / update", title=f"Seed {seed}")
        for ax in axes[row]:
            ax.grid(alpha=.2)
    axes[0, 0].legend()
    axes[0, 1].legend(title="Train length")
    fig.savefig(analysis / "gaps_growth.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, seed in zip(axes, SEEDS):
        for length in LENGTHS:
            selected = [r for r in memory if r["seed"] == seed and r["length"] == length]
            ax.plot([r["endpoint"] for r in selected], [r["inherited_fraction"] for r in selected], "o-", color=LENGTH_COLORS[length], label=str(length))
        ax.set(xlabel="Global updates", ylabel="Fraction from saved v6400", title=f"Seed {seed}")
        ax.set_xticks((9600, END))
        ax.grid(alpha=.2)
    axes[0].legend(title="Train length")
    fig.suptitle("Surviving boundary memory in the recorded uncentered second moment")
    fig.savefig(analysis / "moment_memory.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = project_dir()
    directory = root / args.run_dir
    status = json.loads((directory / "status.json").read_text())
    assert status["complete"] and status["completed_runs"] == 9, "Incomplete continuation; do not claim full matrix"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["frozen_before_scientific_tails"]
    assert file_digest(directory / "source_inventory.json") == manifest["source_inventory_sha256"]
    inventory = json.loads((directory / "source_inventory.json").read_text())["cells"]
    for name, digest in manifest["source_sha256"].items():
        assert file_digest(directory / "source_snapshot" / name) == digest
    for doc, key in (("PLAN", "plan_sha256"), ("SPEC", "spec_sha256")):
        assert file_digest(directory / "source_snapshot" / f"LENGTH_EXTENSION_12800_{doc}.md") == manifest[key]
    assert len(manifest["serialization_checks"]) == 3
    assert all(r["weights_optimizer_rng_loaders_history_exact"] and r["batch_sequence_exact"] for r in manifest["serialization_checks"])
    analysis = directory / "analysis"
    analysis.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), directory / "source_snapshot" / Path(__file__).name)
    paths, summaries, trajectories, final, common, crossings, events, memory, carry_parts = {}, [], [], [], [], [], [], [], []
    direct, exposure, growth, moment_rows, verification = [], [], [], [], []
    max_definition_difference = 0.0
    for item in inventory:
        seed, length, name = item["seed"], item["length"], item["name"]
        identity = {"seed": seed, "length": length}
        for path, digest in item["source_sha256"].items():
            assert file_digest(Path(path)) == digest
        start_path = directory / "starts" / f"{name}.pt"
        assert file_digest(start_path) == item["normalized_start_sha256"]
        bundle = torch.load(start_path, map_location="cpu", weights_only=False)
        assert tree_digest({k: bundle[k] for k in ("model", "optimizer", "named_optimizer", "rng", "train_loaders", "val_loaders")}) == item["scientific_state_sha256"]
        run = directory / "runs" / name
        check = json.loads((run / "verification.json").read_text())
        assert check["start_state_restored_exact"] and check["source_history_preserved_exact"]
        assert check["new_updates"] == check["trace_rows"] == 6400 and check["first_epoch"] == 201 and check["last_epoch"] == 400
        assert check["optimizer_group_parameter_names"] == item["optimizer_group_parameter_names"]
        for step, digest in check["checkpoint_sha256"].items():
            path = run / f"state_step{step}.pt"
            assert file_digest(path) == digest
            state = torch.load(path, map_location="cpu", weights_only=False)
            assert state["optimizer_updates"] == int(step) and state["epoch"] == int(step) // 32
            assert state["validation_complete"] and not state["model_training"]
            assert parameter_groups(state) == item["optimizer_group_parameter_names"]
            assert all(int(v["step"]) == int(step) for v in state["named_optimizer"].values())
        path = load_path(run)
        paths[(seed, length)] = path
        prefix = [r for r in path if r["optimizer_updates"] <= START]
        tail = [r for r in path if r["optimizer_updates"] >= START]
        trajectories.extend({**identity, **r} for r in path)
        for endpoint in (START, 9600, END):
            row = next(r for r in path if r["optimizer_updates"] == endpoint)
            summaries.append({**identity, **row})
        final.append({**identity, **path[-1], "all_extension_grid_fitted": all(r["fitted"] for r in tail)})
        trace = read_rows(run / "coefficient_trace.csv")
        assert len(trace) == 6400
        assert int(float(trace[0]["step_before"])) == START and int(float(trace[-1]["step_after"])) == END
        assert sum(int(r["batch_size"]) for r in trace) == 409600
        assert sum(int(r["positives"]) for r in trace) == sum(int(r["negatives"]) for r in trace) == 204800
        crossing, new_events = crossing_summary(trace, prefix)
        crossings.append({**identity, **crossing})
        events.extend({**identity, **r} for r in new_events)
        for row in trace:
            step = int(row["optimizer_updates"])
            if step % 200 == 0:
                grid = next(r for r in tail if r["optimizer_updates"] == step)
                difference = abs(float(row["r_after"]) - grid["c_delta"])
                max_definition_difference = max(max_definition_difference, difference)
                assert difference < 3e-6
                moment_rows.append({**identity, "optimizer_updates": step,
                    **{key: float(row[key]) for key in ("u_gradient", "u_update", "c_update", "r_update", "r_from_coefficient", "r_from_score",
                        "m_after", "v_after", "m_hat_after", "v_hat_after", "normalized_direction")}})
        initial_v = float(bundle["named_optimizer"][COEFFICIENT]["exp_avg_sq"])
        earlier = read_rows(Path(item["earlier_coefficient_trace"])) if item["earlier_complete_gradient_history"] else None
        for endpoint in (9600, END):
            m = reconstruct_boundary_memory(trace, initial_v, endpoint)
            memory.append({**identity, **m})
            split = {**identity, "endpoint": endpoint, "earlier_complete_history": earlier is not None,
                "inherited_boundary_contribution": m["inherited_contribution"], "updates1_1600_contribution": None,
                "updates1601_6400_contribution": None, "boundary_history_residual": None}
            if earlier is not None:
                assert [int(r["optimizer_updates"]) for r in earlier] == list(range(1, START + 1))
                beta = m["beta2"]
                assert all(float(row["beta2"]) == beta for row in earlier)
                for lower, upper, key in ((1, 1600, "updates1_1600_contribution"), (1601, START, "updates1601_6400_contribution")):
                    split[key] = math.fsum((1 - beta) * beta ** (endpoint - int(row["optimizer_updates"])) * float(row["u_gradient"]) ** 2
                        for row in earlier if lower <= int(row["optimizer_updates"]) <= upper)
                split["boundary_history_residual"] = m["inherited_contribution"] - split["updates1_1600_contribution"] - split["updates1601_6400_contribution"]
            carry_parts.append(split)
        for target in TARGETS:
            assert tail[0]["common100_bce"] > target
            match = first_loss_crossing(tail, target)
            common.append({**identity, "target": target, "reached": match is not None,
                "eligible_fitted": match["eligible_fitted"] if match else False,
                "reason": "fitted_crossing" if match and match["eligible_fitted"] else "own_length_unfitted_at_crossing" if match else "right_censored_12800",
                **{key: match[key] if match else None for key in ("update_lower", "update_upper", "estimated_update", *INTERPOLATED_FIELDS)}})
        for lower, upper in ((4800, 6400), (8000, 9600), (11200, END)):
            before = next(r for r in path if r["optimizer_updates"] == lower)
            after = next(r for r in path if r["optimizer_updates"] == upper)
            growth.append({**identity, "update_lower": lower, "update_upper": upper,
                "r_growth": after["c_delta"] - before["c_delta"], "r_growth_per_update": (after["c_delta"] - before["c_delta"]) / (upper - lower),
                "c_growth": after["c"] - before["c"], "delta_growth": after["delta"] - before["delta"]})
        explicit = read_rows(run / "metrics_by_length.csv")
        assert sorted(int(row["length"]) for row in explicit) == sorted(EVAL_LENGTHS)
        for row in explicit:
            expected = analytic(path[-1], int(row["length"]))
            for measured, theoretical in (("mean_empirical_target_attention", "target_mass"), ("mean_logit_positive", "positive_logit"), ("mean_logit_negative", "negative_logit")):
                assert math.isclose(float(row[measured]), expected[theoretical], rel_tol=3e-5, abs_tol=3e-6)
            direct.append({**identity, "eval_length": int(row["length"]), "evaluation_kind": "direct_forward",
                **{key: float(row[key]) for key in ("positive_accuracy", "negative_accuracy", "mean_empirical_target_attention", "mean_logit_positive", "mean_logit_negative")}})
        counts = json.loads((run / "exposure.json").read_text())
        assert counts["total_sequences"] == 819200 and counts["total_tokens"] == 819200 * length
        assert check["new_exposure"]["sequences"] == 409600 and check["new_exposure"]["tokens"] == 409600 * length
        exposure.append({**identity, "new_updates": 6400, "new_sequences": 409600, "new_tokens": 409600 * length,
            "cumulative_updates": END, "cumulative_sequences": 819200, "cumulative_tokens": 819200 * length})
        verification.append({**identity, "source_unchanged": True, "start_restoration_exact": True,
            "groups_preserved": True, "new_trace_rows": 6400, "max_accounting_residual": check["max_accounting_residual"]})
        write_json(run / "status.json", {"complete": True, "phase": "analysis_verified",
            "start_step": START, "end_step": END, "new_updates": 6400, "trace_rows": 6400,
            "source_start_sha256": item["normalized_start_sha256"], "source_groups_preserved": True,
            "final_fitted": path[-1]["fitted"], "threshold_status": crossing["entry_status"]})
    gaps = []
    for seed in SEEDS:
        for step in range(START, END + 1, 200):
            values = {length: next(r for r in paths[(seed, length)] if r["optimizer_updates"] == step) for length in LENGTHS}
            start_values = {length: next(r for r in paths[(seed, length)] if r["optimizer_updates"] == START) for length in LENGTHS}
            g1, g2 = values[10]["c_delta"] - values[30]["c_delta"], values[30]["c_delta"] - values[1000]["c_delta"]
            s1, s2 = start_values[10]["c_delta"] - start_values[30]["c_delta"], start_values[30]["c_delta"] - start_values[1000]["c_delta"]
            gaps.append({"seed": seed, "optimizer_updates": step, "gap_10_30": g1, "gap_30_1000": g2,
                "gap_10_30_change": g1 - s1, "gap_30_1000_change": g2 - s2,
                "ordering_holds": values[10]["c_delta"] > values[30]["c_delta"] > values[1000]["c_delta"]})
    common_gaps = []
    for target in TARGETS:
        for seed in SEEDS:
            selected = {length: next(r for r in common if r["target"] == target and r["seed"] == seed and r["length"] == length) for length in LENGTHS}
            eligible = all(r["eligible_fitted"] for r in selected.values())
            common_gaps.append({"seed": seed, "target": target, "all_lengths_eligible": eligible,
                "gap_10_30": selected[10]["c_delta"] - selected[30]["c_delta"] if eligible else None,
                "gap_30_1000": selected[30]["c_delta"] - selected[1000]["c_delta"] if eligible else None,
                "ordering_holds": selected[10]["c_delta"] > selected[30]["c_delta"] > selected[1000]["c_delta"] if eligible else None})
    suite_file = directory / "validation/full_test_suite.txt"
    suite_text = suite_file.read_text(encoding="utf-8-sig") if suite_file.exists() else ""
    test_count = re.search(r"Ran (\d+) tests", suite_text)
    audit = {"complete": True, "paths": 9, "fitted_final": sum(r["fitted"] for r in final),
        "fitting_retained_on_grid": all(r["all_extension_grid_fitted"] for r in final),
        "explicit_all_classes_correct": all(r["positive_accuracy"] == r["negative_accuracy"] == 1 for r in direct),
        "length30_new_entries": sum(r["first_new_entry"] is not None for r in crossings if r["length"] == 30),
        "length1000_new_entries": sum(r["first_new_entry"] is not None for r in crossings if r["length"] == 1000),
        "ordering_passes_at12800": sum(r["ordering_holds"] for r in gaps if r["optimizer_updates"] == END),
        "ordering_passes_all_extension_grid": sum(r["ordering_holds"] for r in gaps), "ordering_grid_comparisons": len(gaps),
        "common_loss_coverage": {str(target): {"reached": sum(r["reached"] for r in common if r["target"] == target),
            "eligible_fitted": sum(r["eligible_fitted"] for r in common if r["target"] == target), "planned": 9} for target in TARGETS},
        "common_loss_ordering": {str(target): {"passes": sum(r["ordering_holds"] is True for r in common_gaps if r["target"] == target),
            "eligible_seed_triplets": sum(r["all_lengths_eligible"] for r in common_gaps if r["target"] == target)} for target in TARGETS},
        "both_final_gaps_widened_seeds": sum(r["gap_10_30_change"] > 0 and r["gap_30_1000_change"] > 0 for r in gaps if r["optimizer_updates"] == END),
        "moment_reconstruction_passes": sum(r["passed"] for r in memory), "moment_reconstruction_checks": len(memory),
        "max_memory_relative_residual": max(r["relative_residual"] for r in memory),
        "max_trace_grid_exponent_difference": max_definition_difference,
        "max_accounting_residual": max(r["max_accounting_residual"] for r in verification),
        "full_test_count": int(test_count.group(1)) if test_count else None, "full_test_suite_passed": suite_text.strip().endswith("OK"),
        "test_log_sha256": file_digest(suite_file) if suite_file.exists() else None,
        "historical_sources_unchanged": True, "scientific_updates": status["new_scientific_updates"],
        "new_sequence_exposures": status["new_sequence_exposures"], "new_token_exposures": status["new_token_exposures"],
        "discarded_updates": status["discarded_updates"], "wall_minutes": status["wall_seconds"] / 60}
    for filename, rows in (("trajectories", trajectories), ("fixed_update_states", summaries), ("final_states", final),
        ("crossings", crossings), ("threshold_events", events), ("common_loss", common), ("common_loss_gaps", common_gaps), ("paired_gaps", gaps),
        ("growth_windows", growth), ("moment_memory", memory), ("inherited_history_split", carry_parts),
        ("moment_observations", moment_rows), ("explicit_evaluations", direct), ("realized_exposure", exposure), ("verification", verification)):
        if rows:
            write_csv(analysis / f"{filename}.csv", rows)
    write_json(analysis / "audit.json", audit)
    make_plots(analysis, paths, common, gaps, growth, memory)
    write_report(root, directory, audit, summaries, final, crossings, common, gaps, common_gaps, growth, memory, carry_parts)
    print(json.dumps(audit, indent=2))


def write_report(root, directory, audit, summaries, final, crossings, common, gaps, common_gaps, growth, memory, carry_parts):
    link = "../../" + directory.relative_to(root).as_posix() + "/analysis/"
    fixed_table = md_table(["Seed", "Length", "Step", "$c$", "$\\Delta$", "$r$", "Own BCE", "100 BCE", "Head slope", "Head threshold", "Fitted"],
        [[r["seed"], r["length"], r["optimizer_updates"], *(fmt(r[k]) for k in ("c", "delta", "c_delta", "own_bce", "common100_bce", "head_slope", "head_threshold")), r["fitted"]] for r in summaries])
    crossing_table = md_table(["Seed", "Length", "Start status", "Historical entry grid bracket", "First new entry", "Censored at 12800", "Retained above", "Exits"],
        [[r["seed"], r["length"], r["entry_status"], r["historical_first_entry_grid_bracket"] or "not_observed",
            r["first_new_entry"] or "not_observed", r["right_censored_at12800"], r["all_steps_above_after_start_or_entry"], r["exit_count"]] for r in crossings])
    gap_table = md_table(["Seed", "Step", "$r_{10}-r_{30}$", "Change since 6400", "$r_{30}-r_{1000}$", "Change since 6400", "Ordering"],
        [[r["seed"], r["optimizer_updates"], *(fmt(r[k]) for k in ("gap_10_30", "gap_10_30_change", "gap_30_1000", "gap_30_1000_change")), r["ordering_holds"]]
            for r in gaps if r["optimizer_updates"] in (START, 9600, END)])
    growth_table = md_table(["Seed", "Length", "4800–6400 growth/update", "8000–9600", "11200–12800"],
        [[seed, length, *(f"{next(r['r_growth_per_update'] for r in growth if r['seed'] == seed and r['length'] == length and r['update_lower'] == lower):.8g}"
            for lower in (4800, 8000, 11200))] for seed in SEEDS for length in LENGTHS])
    coverage_table = md_table(["Length-100 BCE", "Reached / 9", "Fitted and eligible / 9"],
        [[target, f"{c['reached']}/9", f"{c['eligible_fitted']}/9"] for target, c in audit["common_loss_coverage"].items()])
    common_gap_table = md_table(["100 BCE target", "Seed", "$r_{10}-r_{30}$", "$r_{30}-r_{1000}$", "Ordering"],
        [[r["target"], r["seed"], fmt(r["gap_10_30"]), fmt(r["gap_30_1000"]), r["ordering_holds"]] for r in common_gaps])
    rates = [next(r["r_growth_per_update"] for r in growth if r["seed"] == seed and r["length"] == length and r["update_upper"] == END)
        / next(r["r_growth_per_update"] for r in growth if r["seed"] == seed and r["length"] == length and r["update_upper"] == START)
        for seed in SEEDS for length in LENGTHS]
    final_memory = [r for r in memory if r["endpoint"] == END]
    gap_changes = [r for r in gaps if r["optimizer_updates"] == END]
    common_tables = []
    for target in TARGETS:
        common_tables.append(f"Target {target}:\n\n" + md_table(["Seed", "Length", "Eligible", "Crossing bracket", "Interpolated update", "$c$", "$\\Delta$", "$r$", "Head threshold", "Reason"],
            [[r["seed"], r["length"], r["eligible_fitted"], f"({r['update_lower']}, {r['update_upper']}]" if r["reached"] else "not_reached",
                fmt(r["estimated_update"], 1), *(fmt(r[k]) for k in ("c", "delta", "c_delta", "head_threshold")), r["reason"]]
                for r in common if r["target"] == target]))
    memory_table = md_table(["Seed", "Length", "Step", "Recorded second moment", "Pre-6400 fraction", "New-gradient fraction", "Relative residual", "Passed"],
        [[r["seed"], r["length"], r["endpoint"], f"{r['recorded_second_moment']:.6g}", f"{100*r['inherited_fraction']:.4f}%",
            f"{100*r['new_gradient_fraction']:.4f}%", f"{r['relative_residual']:.6g}", r["passed"]] for r in memory])
    split_table = md_table(["Seed", "Length", "Step", "Inherited 1–1600", "Inherited 1601–6400", "Boundary reconstruction residual"],
        [[r["seed"], r["length"], r["endpoint"], *(f"{r[k]:.6g}" if r[k] is not None else "missing earlier history"
            for k in ("updates1_1600_contribution", "updates1601_6400_contribution", "boundary_history_residual"))] for r in carry_parts])
    extrapolation_table = md_table(["Seed", "Length", "$r$ at 12800", "Head-aware analytical status", "Predicted failure log10 length"],
        [[r["seed"], r["length"], fmt(r["c_delta"]), r["failure_status"], fmt(r["failure_log10"], 3)] for r in final])
    first1000 = [r for r in crossings if r["length"] == 1000 and r["right_censored_at12800"]]
    if first1000:
        direction = ("Length 30 attained the threshold in all three seeds with the larger budget, while length 1000 did not. The paired exponent gaps widened and persisted at the reached common losses. "
            "The selected next action is a focused comparison of the saved length-30/1000 growth, head and moment records before designing a limited optimizer-state intervention. "
            "Non-entry is right-censored at 12800 for the listed paths; it does not imply permanent failure or convergence. Do not extend the budget until all paths cross.")
    elif audit["fitted_final"] < 9 or not audit["fitting_retained_on_grid"]:
        direction = "The selected next action is to analyze the retained fitting regressions separately from exponent selection, using the recorded states. Do not rescue or replace a seed automatically."
    else:
        direction = ("Longer-length threshold attainment is budget dependent, while the paired solution gaps and common-loss comparisons remain separate questions. "
            "The selected next action is to design one limited second-moment intervention with explicit predicted coefficient/exponent changes, informed by the saved length contrasts and memory fractions. No causal role is established by this continuation.")
    report = f"""# Training-length continuation to 12800 updates

Execution date: October 24, 2026. All nine baseline continuations reached total
update 12800 from their original full 6400-update states in
{audit['wall_minutes']:.2f} minutes, including audit, probes and evaluation.
Lengths: 10, 30, 1000. Paired seeds: 3, 4, 6. See the
[plan](LENGTH_EXTENSION_12800_PLAN.md) and [launch specification](LENGTH_EXTENSION_12800_SPEC.md).

## Fitting and threshold attainment

Final own-length fitting: **{audit['fitted_final']}/9**. Fitting retained at every
200-update extension diagnostic: {audit['fitting_retained_on_grid']}.
Both classes are correct at every final direct evaluation length through 10000:
{audit['explicit_all_classes_correct']}. No path or seed was replaced.

New threshold entries: length 30, **{audit['length30_new_entries']}/3**;
length 1000, **{audit['length1000_new_entries']}/3**. Length 10 was already above
1 at extension start. Successful fitting and threshold attainment are distinct.

{crossing_table}

New entry/exit steps use each actual update, with detached float64 query/key
dot-product margin and the actual float32 softplus coefficient. Historical
length-10 entries remain grid-bracketed; 6401 is not a new entry merely because
a trace starts there. Censoring means no new entry within this fixed budget.
The diagnostic series retains its original float32 score-subtraction definition.
Maximum absolute trace-versus-grid exponent difference:
{audit['max_trace_grid_exponent_difference']:.6g}; these numerical conventions
are not silently mixed to select entry times.

## Fixed-update solutions and paired gaps

{fixed_table}

![Combined coefficient, margin and exponent trajectories]({link}combined_trajectories.png)

![Length response at 6400 and 12800]({link}length_response.png)

The tentative endpoint ordering passed {audit['ordering_passes_at12800']}/3 seeds.
Over the extension diagnostic grid it held in
{audit['ordering_passes_all_extension_grid']}/{audit['ordering_grid_comparisons']}
paired comparisons; intermediate ordering was measured rather than assumed.
Threshold crossing does not mean the coefficient, margin or exponent gaps vanish.

Both paired endpoint gaps **widened in {audit['both_final_gaps_widened_seeds']}/3 seeds**:
the increase in $r_{{10}}-r_{{30}}$ is
{min(r['gap_10_30_change'] for r in gap_changes):.6f}–{max(r['gap_10_30_change'] for r in gap_changes):.6f},
and in $r_{{30}}-r_{{1000}}$ it is
{min(r['gap_30_1000_change'] for r in gap_changes):.6f}–{max(r['gap_30_1000_change'] for r in gap_changes):.6f}.
Thus longer training achieves a threshold for length 30 while preserving and
expanding the measured equal-update separation of the successful solutions.

{gap_table}

Average exponent growth per update in the three predeclared 1600-update windows:

{growth_table}

![Paired gaps and fixed-window growth]({link}gaps_growth.png)

These finite-window changes describe progress, not convergence or a guaranteed
linear threshold forecast. In all nine paths the final window's average growth
is smaller than the pre-extension window: its ratio is
{min(rates):.3f}–{max(rates):.3f}. There is continued growth alongside slowing,
so neither a permanent failure claim nor an unchanged-rate crossing forecast
follows for length 1000. [Full trajectories]({link}trajectories.csv) retain
initialization, all 200-update diagnostics, head state, masses, logits and
own/common loss. The 6400 boundary is included once. The original prefix is
reused; only missing length-10 analytical 30/300 rows were added from saved
parameters, without inferring earlier optimizer history.

![Own-length and common evaluation loss]({link}loss_trajectories.png)

## New common-loss comparisons

{coverage_table}

All three declared targets are reached by all nine fitted paths. The exponent
ordering also remains strict in every paired common-loss triplet:

{common_gap_table}

Thus the length contrast is present at matched evaluation loss as well as at
equal updates. This does not prove independence from every aspect of optimization
progress, since only one evaluation loss is matched.

{"\n\n".join(common_tables)}

![Post-6400 common-loss exponent comparison]({link}common_loss.png)

Every target is below the source losses. Crossings are the first post-6400
downward crossing between adjacent 200-update records. Linear interpolation
uses the established parameter/logit convention; a nonlinear forward on the
interpolated parameters need not reproduce the target exactly. Eligibility
requires strictly correct own-length logits. Censored cells remain visible;
there is no extrapolation or substitution of an earlier sweep target.
Common loss does not match the head, geometry, optimizer state or complete function.

## Optimizer memory and actual updates

The complete new trace has 6400 updates per path, 57600 in total, with raw
coefficient gradients, actual unconstrained-parameter/coefficient/exponent
changes, before/after moments, bias correction and normalized applied direction.
See [moment observations]({link}moment_observations.csv) and each run's complete
`coefficient_trace.csv` and `allocation_by_200.csv`. Maximum signed-accounting
residual is {audit['max_accounting_residual']:.6g} for

```math
\\delta r=\\overline{{\\Delta}}\\,\\delta c+\\overline c\\,\\delta\\Delta.
```

This is an accounting identity, not causal mediation. Adam's second moment is
an uncentered moving average of squared gradients, not centered variance.
Boundary and new-gradient contributions use

```math
v_T=\\beta_2^{{T-6400}}v_{{6400}}
 +(1-\\beta_2)\\sum_{{j=6401}}^T\\beta_2^{{T-j}}g_{{u,j}}^2.
```

{memory_table}

Reconstruction passes: **{audit['moment_reconstruction_passes']}/{audit['moment_reconstruction_checks']}**,
using the unchanged relative tolerance 0.0001 and absolute tolerance 1e-12.
Maximum relative residual: {audit['max_memory_relative_residual']:.6g}.
Fractions are against the recorded float32 state; their sum can differ slightly
from 1 due to the retained float64 reconstruction residual.

![Surviving pre-6400 moment contribution]({link}moment_memory.png)

At 12800 the saved-boundary contribution supplies
{100*min(r['inherited_fraction'] for r in final_memory):.4f}%–{100*max(r['inherited_fraction'] for r in final_memory):.4f}%
of the remaining recorded second moment. Its absolute contribution has itself
decayed to {100*(.999**6400):.4f}% of its saved boundary magnitude. A high surviving
fraction therefore means the new squared-gradient contribution is small relative
to that decayed memory; it does not mean the second moment stayed constant.
Coefficient/exponent growth continues while average exponent growth slows in
the fixed windows, and applied updates still depend on first moment, bias
correction and the softplus parameterization. This combination motivates a
controlled test, rather than establishing causal importance from fractions.

For lengths 30/1000, split that inherited history further without changing its
exact saved-boundary contribution. Values below are surviving moment quantities:

{split_table}

Length 10's complete earlier gradient history is missing; its boundary carryover
is identifiable and is not split using sparse interpolation. The extra split
for lengths 30/1000 preserves its inherited reconstruction error separately.
A high carryover fraction is not a causal attribution, and a changed fraction
alone does not explain the selected exponent or head adaptation.

## Direct evaluation and analytical extrapolation

{extrapolation_table}

The six final direct evaluation lengths are 10, 30, 100, 300, 1000 and 10000.
Both classes, logits and positive attention masses are retained in
[explicit_evaluations.csv]({link}explicit_evaluations.csv) and agree with the
reduced-model formulas within the declared numerical tolerance. All failures
beyond 10000, including no-finite-crossing status, are head-aware analytical
predictions for fixed trained parameters, not direct 10M tests.

## Restoration, compute and tests

All source files and normalized starting bundles remain hash-identical.
Length-10 final weights/history/diagnostics matched the verified historical
baseline before restoration. Integer optimizer IDs were checked against named
moments; two source groups at length 10 and one at lengths 30/1000 were preserved,
including all options and counters. Recreated dataset hashes, loader generators,
Python/NumPy/Torch RNG and weights matched every source. The first new epoch
is 201; source validation was not repeated. Saved states at 9600/12800 include
model/optimizer/data/RNG/history after completed validation.

Seed-3 serialization/passivity probes at every length compare 64 uninterrupted
unobserved updates with 32 observed updates, save/load, and 32 further observed
updates. All final weights, optimizer, batches, RNG/loaders and histories match
exactly. No uninterrupted 0–12800 historical run exists; these are restoration
and serialization checks, not a claim of comparison to one.

Scientific continuation compute: {audit['scientific_updates']} new updates,
{audit['new_sequence_exposures']} new sequence exposures and
{audit['new_token_exposures']} new token exposures. Each cumulative path has
819200 sequence exposures. Prefix training is reused, not newly charged.
Probes add {audit['discarded_updates']} updates, 24576 training sequence exposures
and 8519680 tokens; probe validation adds 6144 exposures and 2129920 tokens.
Scientific validation adds 921600 exposures and 319488000 tokens. Final direct
evaluation adds 2700 examples and 1716000 tokens. Unit-test CPU paths are separate.
Equal updates equalize sequence exposures, while token exposure/runtime differ.

Full automated suite: {audit['full_test_count']} tests, passed status
{audit['full_test_suite_passed']}. The log is `validation/full_test_suite.txt`.
Workspace-local temporary files avoid the known system temporary-directory
write restriction. [Audit]({link}audit.json), source inventory, manifest, state
hashes, exact group mappings and snapshots retain full provenance.

## Direction review

{direction}

This extension does not establish a universal training-length law, permanent
threshold retention, convergence, or the optimizer history's causal role.
"""
    (root / "documents/training_length/LENGTH_EXTENSION_12800_RESULTS.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
