"""Audit behavior-matched coefficient controls and paired factorizations."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_training_length_confirmation import match_loss
from analyze_training_length_followup import crossing, read_csv, trajectory
from run_matched_initial_control import TRACE_STEPS
from stage3_simplified_attention import project_dir, write_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--factorization", action="store_true")
    args = parser.parse_args()
    root = project_dir() / args.run_dir
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=True)
    baseline = project_dir() / "runs/training_length/seed_replication_20260929_01"
    unmatched = project_dir() / "runs/training_length/coefficient_init_20260929_01"
    cohorts = [("matched", root / "main", 30)]
    if args.factorization:
        cohorts.append(("factorization", root / "factorization", 12))
    specs = []
    for seed in range(3, 8):
        for length in (10, 100):
            for tag, directory in (("baseline", baseline), ("c0p1", unmatched), ("c1", unmatched)):
                name = f"learned_log_n{length}_s{seed}" + ("" if tag == "baseline" else f"_{tag}")
                specs.append(({"cohort": "baseline" if tag == "baseline" else "unmatched",
                               "seed": seed, "length": length, "tag": tag,
                               "reference": 0, "factor": "original", "run": name}, directory / name))
    statuses = []
    for cohort, directory, expected in cohorts:
        status = json.loads((directory / "status.json").read_text())
        assert status["complete"] and status["completed_runs"] == expected
        statuses.append(status)
        rows = read_csv(directory / "summary.csv")
        assert len(rows) == expected
        for row in rows:
            specs.append(({"cohort": cohort, "seed": int(row["seed"]), "length": int(row["train_lengths"]),
                           "tag": "c0p1" if "c0p1" in row["run"] else "c1",
                           "reference": int(row["reference_length"]), "factor": row["factorization"],
                           "run": row["run"]}, directory / row["run"]))
    trajectories, records, matching, first_updates, windows, signs, norm_records = {}, [], [], [], [], [], []
    trace_lookup = {}
    for meta, run in specs:
        key = (meta["cohort"], meta["seed"], meta["length"], meta["tag"], meta["reference"], meta["factor"])
        diag = trajectory(run)
        trajectories[key] = diag
        raw = read_csv(run / "training_diagnostics.csv")
        own = {int(row["optimizer_updates"]): row for row in raw if int(row["length"]) == meta["length"]}
        final = diag[6400]
        explicit = read_csv(run / "metrics_by_length.csv")
        exposure = json.loads((run / "exposure.json").read_text())
        assert exposure["total_sequences"] == 409600
        assert sum(item["positives"] for item in exposure["by_length"].values()) == 204800
        assert sum(item["negatives"] for item in exposure["by_length"].values()) == 204800
        assert exposure["total_tokens"] == 409600 * meta["length"]
        row = {**meta, "path": str(run.relative_to(project_dir())),
               "initial_c": diag[0]["c"], "initial_delta": diag[0]["delta"],
               "initial_common100_bce": diag[0]["common100_bce"],
               "initial_own_bce": float(own[0]["balanced_bce"]),
               **{f"final_{field}": final[field] for field in
                  ("c", "delta", "c_delta", "head_slope", "head_intercept", "head_threshold",
                   "common100_bce", "training_classes_correct", "failure_status", "failure_log10")},
               "final_own_bce": float(own[6400]["balanced_bce"]), **crossing(diag),
               "first_recorded_fitting": next((step for step, item in diag.items()
                                                if item["training_classes_correct"]), None),
               "min_explicit_accuracy": min(float(item["accuracy"]) for item in explicit),
               "total_tokens": exposure["total_tokens"]}
        records.append(row)
        if meta["cohort"] not in {"matched", "factorization"}:
            continue
        check = json.loads((run / "initial_matching.json").read_text())
        assert check["rng_unchanged"] and check["head_bitwise_equal"]
        previous_config = json.loads((baseline / f"learned_log_n{meta['length']}_s{meta['seed']}" / "config.json").read_text())
        config = json.loads((run / "config.json").read_text())
        assert previous_config["initial_state_sha256"] == check["baseline_initial_sha256"]
        assert config["initial_state_sha256"] == check["matched_initial_sha256"]
        matching.append({**meta, "requested_coefficient": 0.1 if meta["tag"] == "c0p1" else 1.0,
                         "ratio": check["ratio"], "query_scale": check["query_scale"],
                         "key_scale": check["key_scale"],
                         "max_reference_function_residual": max(check["reference_function_residuals"].values()),
                         "max_raw_score_scaling_residual": check["max_raw_score_scaling_residual"],
                         "max_factor_function_residual": max(check["factorization_function_residuals"].values())
                         if check["factorization_function_residuals"] else None,
                         **{f"{which}_{field}": check[which][field] for which in ("baseline", "matched")
                            for field in ("c", "delta", "c_delta", "query_norm", "key_norm",
                                          "effective_margin_10", "effective_margin_100",
                                          "target_mass_10", "target_mass_100",
                                          "balanced_bce_10", "balanced_bce_100")}})
        updates = read_csv(run / "update_trace.csv")
        assert {int(item["optimizer_updates"]) for item in updates} == TRACE_STEPS
        assert len(updates) == 242
        trace_lookup[key] = updates
        first_updates.append({**meta, **{field: float(value) for field, value in updates[0].items()
                                             if field not in {"q_before", "q_after", "kd_before", "kd_after",
                                                              "q_gradient", "key_gradient", "head_weight_gradient",
                                                              "q_update", "kd_update", "query_full_update", "key_full_update"}
                                             and value != ""}})
        for name, low, high in (("1-200", 1, 200), ("1590-1610", 1590, 1610), ("3190-3210", 3190, 3210)):
            selected = [item for item in updates if low <= int(item["optimizer_updates"]) <= high]
            windows.append({**meta, "window": name, "steps": len(selected),
                **{field: sum(float(item[field]) for item in selected) for field in
                   ("c_update", "delta_update", "delta_from_query_update", "delta_from_key_update",
                    "c_delta_from_c_update", "c_delta_from_score_update",
                    "effective_margin_10_from_c_update", "effective_margin_10_from_score_update",
                    "effective_margin_100_from_c_update", "effective_margin_100_from_score_update")},
                **{f"mean_{field}": statistics.mean(float(item[field]) for item in selected) for field in
                   ("u_gradient", "u_update", "query_full_update_norm", "key_full_update_norm",
                    "query_relative_update_norm", "key_relative_update_norm")}})
        early = [item for item in updates if int(item["optimizer_updates"]) <= 200]
        observed = {step: item for step, item in diag.items()}
        observed.update({int(item["optimizer_updates"]): {field: float(item[f"{field}_after"])
                                                          for field in ("head_slope", "delta")} for item in updates})
        signs.append({**meta,
                      **{f"first_positive_{field}_by_200": next((int(item["optimizer_updates"]) for item in early
                                                                  if float(item[f"{field}_after"]) > 0), None)
                         for field in ("head_slope", "delta")},
                      **{f"first_observed_positive_{field}": next((step for step in sorted(observed)
                                                                   if observed[step][field] > 0), None)
                         for field in ("head_slope", "delta")}})
        norm_records.extend({**meta, **item} for item in read_csv(run / "norm_diagnostics.csv"))
    historical_first = []
    old_trace_dir = project_dir() / "runs/training_length/update_trace_20260929_01"
    for seed in range(3, 6):
        for length in (10, 100):
            for tag in ("baseline", "c0p1", "c1"):
                path = old_trace_dir / f"learned_log_n{length}_s{seed}_{tag}" / "update_trace.csv"
                item = read_csv(path)[0]
                historical_first.append({"seed": seed, "length": length, "tag": tag,
                    **{field: float(item[field]) for field in
                       ("u_gradient", "u_update", "c_update", "delta_update", "c_delta_update",
                        "q_update_norm", "kd_update_norm", "q_gradient_norm", "key_gradient_norm", "head_slope_before")},
                    **{f"own_{field}": float(item[f"{field.split('_before')[0]}_{length}_before"])
                       for field in ("effective_margin_before", "target_mass_before", "positive_logit_before")},
                    "own_effective_margin_update": float(item[f"effective_margin_{length}_update"])})
    first_pairs = []
    for row in first_updates:
        if row["cohort"] != "matched" or row["reference"] != row["length"] or row["seed"] > 5:
            continue
        old = next(item for item in historical_first if item["tag"] == "baseline"
                   and item["seed"] == row["seed"] and item["length"] == row["length"])
        first_pairs.append({key: row[key] for key in ("seed", "length", "tag", "reference", "factor")}
                           | {f"{field}_ratio_to_baseline": row[field] / old[field] if old[field] != 0 else None
                              for field in ("u_gradient", "u_update", "c_update", "delta_update", "q_gradient_norm",
                                            "key_gradient_norm", "q_update_norm")}
                           | {"own_effective_margin_update_ratio_to_baseline":
                              row[f"effective_margin_{row['length']}_update"] / old["own_effective_margin_update"]})
    for filename, rows in (("per_run.csv", records), ("initial_matching.csv", matching),
                           ("first_updates.csv", first_updates), ("trace_windows.csv", windows),
                           ("early_signs.csv", signs), ("norm_trajectories.csv", norm_records),
                           ("historical_first_updates.csv", historical_first), ("first_update_pairs.csv", first_pairs)):
        write_csv(analysis / filename, rows)
    groups = defaultdict(list)
    for row in records:
        groups[(row["cohort"], row["length"], row["tag"], row["reference"], row["factor"])].append(row)
    grouped = []
    for (cohort, length, tag, reference, factor), members in groups.items():
        fitted = [row for row in members if row["final_training_classes_correct"]]
        grouped.append({"cohort": cohort, "length": length, "tag": tag, "reference": reference,
                        "factor": factor, "n": len(members), "fitted": len(fitted),
                        "failed_seeds": " ".join(str(row["seed"]) for row in members if not row["final_training_classes_correct"]),
                        "crossed": sum(row["first_above"] is not None for row in members),
                        **{f"mean_{field}": statistics.mean(row[field] for row in members) for field in
                           ("final_c", "final_delta", "final_c_delta", "final_own_bce", "final_common100_bce")},
                        "sd_final_c_delta": statistics.stdev(row["final_c_delta"] for row in members),
                        "fitted_mean_final_c_delta": statistics.mean(row["final_c_delta"] for row in fitted) if fitted else None})
    write_csv(analysis / "group_summary.csv", grouped)
    matched, parameter_differences, length_pairs, factor_pairs = [], [], [], []
    for meta, _ in specs:
        key = (meta["cohort"], meta["seed"], meta["length"], meta["tag"], meta["reference"], meta["factor"])
        diag = trajectories[key]
        if meta["cohort"] == "baseline":
            continue
        base = trajectories[("baseline", meta["seed"], meta["length"], "baseline", 0, "original")]
        for step in (200, 1600, 3200, 6400):
            parameter_differences.append({**meta, "optimizer_updates": step,
                **{f"{field}_minus_baseline": diag[step][field] - base[step][field] for field in
                   ("c", "delta", "c_delta", "common100_bce")}})
        for target in (0.04, 0.02, 0.01, 0.005):
            a = match_loss(list(base.values()), "common100_bce", target)
            b = match_loss(list(diag.values()), "common100_bce", target)
            matched.append({**meta, "comparison": "within_length_vs_baseline", "target_common100_bce": target,
                            "both_reached": a is not None and b is not None,
                            **{f"{field}_difference": b[field] - a[field] if a and b else None for field in
                               ("estimated_update", "c", "delta", "c_delta", "head_threshold")}})
    for seed in range(3, 8):
        for tag in ("baseline", "c0p1", "c1"):
            regimes = [("baseline" if tag == "baseline" else "unmatched", 0, 0, "original")]
            if tag != "baseline":
                regimes += [("matched", 100, 100, "symmetric"), ("matched", 10, 100, "symmetric")]
            for cohort, ref10, ref100, factor in regimes:
                a = trajectories[(cohort, seed, 10, tag, ref10, factor)]
                b = trajectories[(cohort, seed, 100, tag, ref100, factor)]
                for step in (200, 1600, 3200, 6400):
                    length_pairs.append({"seed": seed, "tag": tag, "cohort": cohort, "reference_10": ref10,
                        "reference_100": ref100, "optimizer_updates": step,
                        "both_fitted": a[step]["training_classes_correct"] and b[step]["training_classes_correct"],
                        **{f"{field}_10_minus_100": a[step][field] - b[step][field] for field in
                           ("c", "delta", "c_delta", "common100_bce")}})
                for target in (0.04, 0.02, 0.01, 0.005):
                    x = match_loss(list(a.values()), "common100_bce", target)
                    y = match_loss(list(b.values()), "common100_bce", target)
                    matched.append({"cohort": cohort, "seed": seed, "length": "10_vs_100", "tag": tag,
                        "reference": f"{ref10}/{ref100}", "factor": factor, "comparison": "length10_minus_length100",
                        "target_common100_bce": target, "both_reached": x is not None and y is not None,
                        **{f"{field}_difference": x[field] - y[field] if x and y else None for field in
                           ("estimated_update", "c", "delta", "c_delta", "head_threshold")}})
    if args.factorization:
        for seed in range(3, 6):
            for length in (10, 100):
                symkey = ("matched", seed, length, "c1", length, "symmetric")
                sym = trajectories[symkey]
                first = trace_lookup[symkey][0]
                for factor in ("query_only", "key_only"):
                    key = ("factorization", seed, length, "c1", length, factor)
                    diag, update = trajectories[key], trace_lookup[key][0]
                    factor_pairs.append({"seed": seed, "length": length, "factor": factor,
                        "symmetric_fitted": sym[6400]["training_classes_correct"],
                        "factor_fitted": diag[6400]["training_classes_correct"],
                        **{f"final_{field}_minus_symmetric": diag[6400][field] - sym[6400][field] for field in
                           ("c", "delta", "c_delta", "common100_bce")},
                        **{f"first_{field}_minus_symmetric": float(update[field]) - float(first[field]) for field in
                           ("u_gradient", "u_update", "delta_update", "c_delta_update", "query_full_update_norm",
                            "key_full_update_norm")},
                        "first_own_effective_margin_update_minus_symmetric":
                        float(update[f"effective_margin_{length}_update"]) - float(first[f"effective_margin_{length}_update"])})
    for filename, rows in (("parameters_vs_baseline.csv", parameter_differences), ("loss_matched.csv", matched),
                           ("length_pairs.csv", length_pairs), ("factorization_pairs.csv", factor_pairs)):
        if rows:
            write_csv(analysis / filename, rows)
    audit = {"analyzed_runs": len(records), "new_runs": len(matching), "training_statuses": statuses,
             "max_reference_residual": max(row["max_reference_function_residual"] for row in matching),
             "max_factor_function_residual": max((row["max_factor_function_residual"] or 0 for row in matching)),
             "new_trace_rows": 242 * len(matching), "max_decomposition_residual": max(
                 abs(float(item[field])) for updates in trace_lookup.values() for item in updates
                 for field in ("delta_decomposition_residual", "c_delta_decomposition_residual",
                               "effective_margin_10_decomposition_residual", "effective_margin_100_decomposition_residual")),
             "baseline_verification": [json.loads(path.read_text()) for path in
                 (root / "main").glob("baseline_verification_*/baseline_verification.json")]}
    assert len(audit["baseline_verification"]) == 2
    assert all(item["weights_bitwise_equal"] and item["history_exact"] and item["diagnostics_exact"]
               for item in audit["baseline_verification"])
    write_json(analysis / "audit.json", audit)
    colors = {"baseline": "#666666", "c0p1": "#0072B2", "c1": "#D55E00"}
    labels = {"baseline": "baseline", "c0p1": "$c_0=0.1$", "c1": "$c_0=1$"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for column, length in enumerate((10, 100)):
        for tag in ("baseline", "c0p1", "c1"):
            for seed in range(3, 8):
                key = ("baseline", seed, length, tag, 0, "original") if tag == "baseline" else (
                    "matched", seed, length, tag, length, "symmetric")
                diag = trajectories[key]
                axes[0, column].plot(list(diag), [item["c_delta"] for item in diag.values()], color=colors[tag],
                                     alpha=0.45, label=labels[tag] if seed == 3 else None)
            seed5key = ("baseline", 5, length, tag, 0, "original") if tag == "baseline" else (
                "matched", 5, length, tag, length, "symmetric")
            seed5 = trajectories[seed5key]
            axes[1, column].plot(list(seed5), [item["common100_bce"] for item in seed5.values()],
                                 color=colors[tag], label=labels[tag])
        axes[0, column].axhline(1, color="black", linewidth=1, linestyle="--")
        axes[0, column].set(title=f"Train/reference {length}: all five seeds", ylabel="Exponent cΔ")
        axes[1, column].set(title=f"Seed 5: train/reference {length}", ylabel="Common length-100 BCE", yscale="log")
    for axis in axes.flat:
        axis.set_xlabel("Optimizer updates")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(analysis / "matched_trajectories.png", dpi=170)
    plt.close(fig)
    if args.factorization:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for axis, length in zip(axes, (10, 100)):
            for factor, color in (("symmetric", "#0072B2"), ("query_only", "#D55E00"), ("key_only", "#009E73")):
                for seed in range(3, 6):
                    diag = trajectories[("matched" if factor == "symmetric" else "factorization",
                                          seed, length, "c1", length, factor)]
                    axis.plot(list(diag), [item["c_delta"] for item in diag.values()], color=color,
                              alpha=0.65, linestyle="--" if seed == 5 else "-",
                              label=factor.replace("_", " ") if seed == 3 else None)
            axis.axhline(1, color="black", linewidth=1, linestyle=":")
            axis.set(title=f"Train/reference {length}; dashed = seed 5", xlabel="Updates", ylabel="Exponent cΔ")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(analysis / "factorization_trajectories.png", dpi=170)
        plt.close(fig)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
