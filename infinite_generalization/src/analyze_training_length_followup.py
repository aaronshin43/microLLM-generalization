"""Audit seed replication, coefficient intervention, and realized-update traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_training_length_confirmation import failure_status, match_loss
from stage3_simplified_attention import project_dir, write_csv, write_json


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def trajectory(run):
    rows = read_csv(run / "training_diagnostics.csv")
    config = json.loads((run / "config.json").read_text())
    train_lengths = config["train_lengths"]
    steps = sorted({int(row["optimizer_updates"]) for row in rows})
    assert steps == list(range(0, 6401, 200)) and len(rows) == 132
    by_step = {}
    for step in steps:
        item = next(row for row in rows if int(row["optimizer_updates"]) == step and int(row["length"]) == 100)
        values = {key: float(item[key]) if item[key] else None for key in
                  ("c", "delta", "c_delta", "head_slope", "head_intercept", "head_threshold", "balanced_bce")}
        values.update(optimizer_updates=step, common100_bce=float(item["balanced_bce"]))
        own_rows = [row for row in rows if int(row["optimizer_updates"]) == step
                    and int(row["length"]) in train_lengths]
        values["training_classes_correct"] = (len(own_rows) == len(train_lengths)
            and all(float(row["positive_logit"]) >= 0 and float(row["negative_logit"]) < 0
                    for row in own_rows))
        values["failure_status"], values["failure_log10"] = failure_status(values)
        by_step[step] = values
    return by_step


def crossing(rows):
    steps = sorted(rows)
    above = [step for step in steps if rows[step]["c_delta"] > 1]
    first = above[0] if above else None
    later_below = [step for step in steps if first is not None and step > first and rows[step]["c_delta"] <= 1]
    return {"first_above": first,
            "previous_recording": steps[steps.index(first) - 1] if first is not None and first > 0 else None,
            "recrossings_below": " ".join(map(str, later_below)),
            "above_through_6400": bool(first is not None and not later_below),
            "right_censored_at_6400": first is None}


def rank(trajectories, seed, step):
    items = []
    for condition in ("10", "100", "10 100"):
        values = trajectories[(seed, condition)][step]
        if not values["training_classes_correct"] or values["failure_status"] not in {"finite", "no_finite_crossing"}:
            return "head_or_prediction_not_comparable"
        items.append((condition, math.inf if values["failure_status"] == "no_finite_crossing"
                      else values["failure_log10"]))
    ordered = sorted(items, key=lambda item: item[1], reverse=True)
    return " > ".join(f"[{condition.replace(' ', ',')}]" for condition, _ in ordered)


def fmt(values, digits=4):
    return f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-dir", required=True)
    parser.add_argument("--replication-dir", required=True)
    parser.add_argument("--intervention-dir", required=True)
    parser.add_argument("--trace-dir", required=True)
    args = parser.parse_args()
    old = project_dir() / args.old_dir
    replication = project_dir() / args.replication_dir
    intervention = project_dir() / args.intervention_dir
    traces = project_dir() / args.trace_dir
    for directory, expected in ((replication, 15), (intervention, 20), (traces, 18)):
        state = json.loads((directory / "status.json").read_text())
        assert state["complete"] and state["completed_runs"] == expected
    analysis = replication / "analysis"
    analysis.mkdir(exist_ok=True)
    old_summary = read_csv(old / "pilot_summary.csv")
    new_summary = read_csv(replication / "pilot_summary.csv")
    init_summary = read_csv(intervention / "intervention_summary.csv")
    assert len(old_summary) == 9 and len(new_summary) == 15 and len(init_summary) == 20
    initial_states = read_csv(intervention / "initial_states.csv")
    assert len(initial_states) == 30
    for seed in range(3, 8):
        states = [row for row in initial_states if int(row["seed"]) == seed]
        assert len(states) == 6 and len({row["noncoefficient_state_sha256"] for row in states}) == 1
    trajectories = {}
    replicated = []
    for summary, directory, cohort in ((old_summary, old, "original"), (new_summary, replication, "new")):
        for row in summary:
            seed, condition = int(row["seed"]), row["train_lengths"]
            run = directory / row["run"]
            diag = trajectory(run)
            trajectories[(seed, condition)] = diag
            exposure = json.loads((run / "exposure.json").read_text())
            assert exposure["total_sequences"] == 409600
            for length, counts in exposure["by_length"].items():
                expected = 204800 if condition == "10 100" else 409600
                assert counts["sequences"] == expected and counts["positives"] == counts["negatives"] == expected // 2
            replicated.append({"seed": seed, "cohort": cohort, "train_lengths": condition,
                               **{key: diag[6400][key] for key in ("c", "delta", "c_delta", "head_threshold",
                                                                   "failure_status", "failure_log10", "training_classes_correct")},
                               **crossing(diag), "total_sequences": exposure["total_sequences"],
                               "total_tokens": exposure["total_tokens"]})
    write_csv(analysis / "replication_per_run.csv", replicated)
    paired_replication = []
    for seed in range(8):
        single10 = trajectories[(seed, "10")][6400]
        single100 = trajectories[(seed, "100")][6400]
        mixed = trajectories[(seed, "10 100")][6400]
        paired_replication.append({"seed": seed, "cohort": "original" if seed < 3 else "new",
            "c_delta_10_minus_100": single10["c_delta"] - single100["c_delta"],
            "c_delta_10_minus_mixed": single10["c_delta"] - mixed["c_delta"],
            "c_delta_mixed_minus_100": mixed["c_delta"] - single100["c_delta"],
            "delta_mixed_minus_100": mixed["delta"] - single100["delta"],
            "failure_rank_1600": rank(trajectories, seed, 1600),
            "failure_rank_6400": rank(trajectories, seed, 6400)})
    write_csv(analysis / "replication_paired.csv", paired_replication)
    init_trajectories = {}
    intervention_rows = []
    for seed in range(3, 8):
        for tag, directory in (("baseline", replication), ("c0p1", intervention), ("c1", intervention)):
            for condition in ("10", "100"):
                run_name = f"learned_log_n{condition}_s{seed}" + ("" if tag == "baseline" else f"_{tag}")
                run = directory / run_name
                diag = trajectory(run)
                init_trajectories[(seed, tag, condition)] = diag
                config = json.loads((run / "config.json").read_text())
                exposure = json.loads((run / "exposure.json").read_text())
                assert exposure["total_sequences"] == 409600
                intervention_rows.append({"seed": seed, "initial_tag": tag, "train_lengths": condition,
                    "initial_u": config["alpha_log_scale_init"], "initial_c": diag[0]["c"],
                    "initial_delta": diag[0]["delta"], "initial_effective_margin_100":
                    (1 + diag[0]["c"] * math.log1p(100)) * diag[0]["delta"],
                    "final_c": diag[6400]["c"], "final_delta": diag[6400]["delta"],
                    "final_c_delta": diag[6400]["c_delta"], "final_common100_bce": diag[6400]["common100_bce"],
                    "final_failure_status": diag[6400]["failure_status"],
                    "final_training_classes_correct": diag[6400]["training_classes_correct"],
                    **crossing(diag),
                    "total_sequences": exposure["total_sequences"], "total_tokens": exposure["total_tokens"]})
    write_csv(analysis / "intervention_per_run.csv", intervention_rows)
    intervention_paired = []
    for seed in range(3, 8):
        for tag in ("baseline", "c0p1", "c1"):
            ten, hundred = (init_trajectories[(seed, tag, condition)][6400] for condition in ("10", "100"))
            intervention_paired.append({"seed": seed, "initial_tag": tag,
                "c_delta_10_minus_100": ten["c_delta"] - hundred["c_delta"],
                "c_10_minus_100": ten["c"] - hundred["c"],
                "delta_10_minus_100": ten["delta"] - hundred["delta"],
                "common100_bce_10_minus_100": ten["common100_bce"] - hundred["common100_bce"]})
    write_csv(analysis / "intervention_paired.csv", intervention_paired)
    matched = []
    for seed in range(3, 8):
        for tag in ("baseline", "c0p1", "c1"):
            ten = list(init_trajectories[(seed, tag, "10")].values())
            hundred = list(init_trajectories[(seed, tag, "100")].values())
            for target in (0.04, 0.02, 0.01, 0.005):
                x, y = match_loss(ten, "common100_bce", target), match_loss(hundred, "common100_bce", target)
                matched.append({"seed": seed, "initial_tag": tag, "target_bce": target,
                    "both_reached": x is not None and y is not None,
                    "c_delta_10_minus_100": x["c_delta"] - y["c_delta"] if x and y else None,
                    "delta_10_minus_100": x["delta"] - y["delta"] if x and y else None,
                    "estimated_update_10_minus_100": x["estimated_update"] - y["estimated_update"] if x and y else None})
    write_csv(analysis / "intervention_loss_matched.csv", matched)
    trace_summary = read_csv(traces / "trace_summary.csv")
    assert len(trace_summary) == 18
    traced_rows = []
    windows = []
    for row in trace_summary:
        run = traces / row["run"]
        verify = json.loads((run / "trace_verification.json").read_text())
        assert verify["weights_bitwise_equal"] and verify["diagnostics_exact"] and verify["history_exact"]
        assert max(verify["max_decomposition_residuals"].values()) < 1e-10
        updates = read_csv(run / "update_trace.csv")
        assert len(updates) == int(row["traced_updates"])
        seed, length = int(row["seed"]), row["train_lengths"]
        tag = next(tag for tag in ("baseline", "c0p1", "c1") if row["run"].endswith(f"_{tag}"))
        for item in updates:
            step = int(item["optimizer_updates"])
            if step <= 200:
                window = "1-200"
            elif 1590 <= step <= 1610:
                window = "1590-1610"
            elif 3190 <= step <= 3210:
                window = "3190-3210"
            else:
                window = "threshold_neighborhood"
            traced_rows.append({"seed": seed, "initial_tag": tag, "train_lengths": length,
                                "optimizer_updates": step, "window": window,
                                **{key: float(item[key]) for key in
                                   ("u_gradient", "u_update", "c_update", "delta_update",
                                    "delta_from_query_update", "delta_from_key_update",
                                    "c_delta_from_c_update", "c_delta_from_score_update",
                                    "effective_margin_10_from_c_update", "effective_margin_10_from_score_update",
                                    "effective_margin_100_from_c_update", "effective_margin_100_from_score_update",
                                    "target_mass_10_before", "target_mass_100_before",
                                    "head_slope_after", "delta_after")}})
    write_csv(analysis / "trace_update_summary.csv", traced_rows)
    transitions = []
    for seed in range(3, 6):
        for tag in ("baseline", "c0p1", "c1"):
            for condition in ("10", "100"):
                early = [row for row in traced_rows if row["seed"] == seed and row["initial_tag"] == tag
                         and row["train_lengths"] == condition and row["optimizer_updates"] <= 200]
                transitions.append({"seed": seed, "initial_tag": tag, "train_lengths": condition,
                    "initial_target_mass_100": early[0]["target_mass_100_before"],
                    "first_positive_head_slope_by_200": next((row["optimizer_updates"] for row in early
                        if row["head_slope_after"] > 0), None),
                    "first_positive_delta_by_200": next((row["optimizer_updates"] for row in early
                        if row["delta_after"] > 0), None)})
    write_csv(analysis / "trace_early_transitions.csv", transitions)
    for seed in range(3, 6):
        for tag in ("baseline", "c0p1", "c1"):
            for condition in ("10", "100"):
                for window in ("1-200", "1590-1610", "3190-3210"):
                    members = [row for row in traced_rows if row["seed"] == seed and row["initial_tag"] == tag
                               and row["train_lengths"] == condition and row["window"] == window]
                    assert len(members) == (200 if window == "1-200" else 21)
                    windows.append({"seed": seed, "initial_tag": tag, "train_lengths": condition, "window": window,
                        "steps": len(members), **{key: sum(row[key] for row in members) for key in
                        ("c_update", "delta_update", "delta_from_query_update", "delta_from_key_update",
                         "c_delta_from_c_update", "c_delta_from_score_update",
                         "effective_margin_100_from_c_update", "effective_margin_100_from_score_update")},
                        "mean_u_gradient": statistics.mean(row["u_gradient"] for row in members),
                        "mean_u_update": statistics.mean(row["u_update"] for row in members),
                        "start_target_mass_100": members[0]["target_mass_100_before"]})
    write_csv(analysis / "trace_window_summary.csv", windows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"10": "#0072B2", "100": "#D55E00", "10 100": "#009E73"}
    for condition in ("10", "100", "10 100"):
        for seed in range(3, 8):
            rows = trajectories[(seed, condition)]
            axes[0].plot(list(rows), [item["c_delta"] for item in rows.values()],
                         color=colors[condition], alpha=0.45,
                         label=f"[{condition.replace(' ', ',')}]" if seed == 3 else None)
    axes[0].axhline(1, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Optimizer updates", ylabel="Exponent cΔ", title="New seeds 3–7")
    for tag, x in (("baseline", 0.006715), ("c0p1", 0.1), ("c1", 1.0)):
        for condition in ("10", "100"):
            values = []
            for seed in range(3, 8):
                final = init_trajectories[(seed, tag, condition)][6400]
                values.append(final["c_delta"])
                axes[1].scatter(x, final["c_delta"], color=colors[condition], alpha=0.65,
                                marker="o" if final["training_classes_correct"] else "x",
                                label=f"[{condition}]" if tag == "baseline" and seed == 3 else None)
            axes[1].plot(x, statistics.mean(values), marker="D", markersize=9, color=colors[condition])
    axes[1].set_xscale("log")
    axes[1].set(xlabel="Initial coefficient c", ylabel="Final exponent cΔ",
                title="Initialization (× failed fit; ◆ mean of all five)")
    axes[0].legend()
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(analysis / "followup_overview.png", dpi=170)
    plt.close(fig)
    write_json(analysis / "audit.json", {"replication_runs": 15, "intervention_new_runs": 20,
        "traced_replays": 18, "trace_rows": len(traced_rows),
        "max_trace_decomposition_residual": max(float(row["max_decomposition_residual"]) for row in trace_summary)})
    shutil.copy2(__file__, analysis / "analyze_training_length_followup.py")
    print("Replication paired:")
    for row in paired_replication:
        print(row)
    print("Intervention paired:")
    for row in intervention_paired:
        print(row)
    print(f"Analysis: {analysis}")


if __name__ == "__main__":
    main()
