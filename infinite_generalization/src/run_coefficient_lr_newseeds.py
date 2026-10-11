"""Validate a frozen length-specific intervention rule on independent seeds."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import torch

from analyze_coefficient_lr_branches import interpolate_loss
from analyze_coefficient_lr_holdout import load_trajectory
from run_coefficient_lr_branches import (
    BranchTrace, PREFIX_STEPS, TAIL_STEPS, capture_parent, capture_rng, continue_training,
    cpu_tree, loader_state, named_optimizer_state, read_rows, restored_recorder,
    short_parent_checks, tree_digest,
)
from run_coefficient_lr_holdout import file_digest
from run_matched_initial_control import fixed_inputs, initial_metrics
from run_training_length_pilot import TrainingRecorder, finish_run
from stage3_simplified_attention import Stage3Config, project_dir, resolve_device, set_reproducibility, train_model, write_csv, write_json

SEEDS = (8, 9)
LENGTHS = (10, 100)
COMPARISONS = ("update6400", "common100_bce0.01")
RELATIVE_TOLERANCE = 0.2


def calibrate_rule(original, holdout):
    source_paths = (original / "analysis/trajectories.csv", holdout / "analysis/numeric_predictions.csv")
    parents = {(int(row["seed"]), int(row["length"])): float(row["c_delta"])
        for row in read_rows(source_paths[0]) if row["tag"] == "control" and int(row["optimizer_updates"]) == 1600}
    ratios = []
    for row in read_rows(source_paths[1]):
        seed, length = int(row["seed"]), int(row["length"])
        assert seed in (3, 4, 6) and row["reached"] == "True"
        baseline_growth = float(row["control_r"]) - parents[(seed, length)]
        assert baseline_growth > 0
        ratios.append({"seed": seed, "length": length, "comparison": row["comparison"],
            "parent_r": parents[(seed, length)], "control_r": float(row["control_r"]),
            "control_growth": baseline_growth, "observed_effect": float(row["actual_effect"]),
            "normalized_effect": float(row["actual_effect"]) / baseline_growth})
    assert len(ratios) == 12
    coefficients = [{"length": length, "comparison": comparison, "old_seeds": [3, 4, 6],
        "normalized_effect_mean": statistics.mean(row["normalized_effect"] for row in ratios
            if row["length"] == length and row["comparison"] == comparison)}
        for length in LENGTHS for comparison in COMPARISONS]
    return {"rule": "r2_hat = r1 + mean_old_normalized_effect[length,comparison] * (r1-r0)",
        "training_seeds": [3, 4, 6], "validation_seeds": list(SEEDS),
        "comparisons": list(COMPARISONS), "relative_effect_tolerance": RELATIVE_TOLERANCE,
        "coefficients": coefficients, "calibration_ratios": ratios,
        "source_sha256": {str(path): file_digest(path) for path in source_paths},
        "new_seed_refitting_allowed": False, "coefficient_rates": [1, 2]}


def predict_from_control(rule, seed, length, parent_r, control, eligible):
    result = []
    for comparison in COMPARISONS:
        coefficient = next(row["normalized_effect_mean"] for row in rule["coefficients"]
            if row["length"] == length and row["comparison"] == comparison)
        values = control[-1] if comparison == "update6400" else interpolate_loss(control, "common100_bce", 0.01)
        control_r = values["c_delta"] if values is not None else None
        growth = control_r - parent_r if control_r is not None else None
        evaluable = eligible and growth is not None and growth > 0
        effect = coefficient * growth if evaluable else None
        reason = "eligible_positive_control_growth" if evaluable else (
            "parent_not_fitted_at1600" if not eligible else "control_target_censored" if control_r is None else "nonpositive_control_growth")
        result.append({"seed": seed, "length": length, "comparison": comparison,
            "eligible_parent": eligible, "evaluable": evaluable, "reason": reason,
            "frozen_coefficient": coefficient, "parent_r": parent_r, "control_r": control_r,
            "control_growth": growth, "predicted_effect": effect,
            "predicted_r": control_r + effect if evaluable else None,
            "effect_lower": (1 - RELATIVE_TOLERANCE) * effect if evaluable else None,
            "effect_upper": (1 + RELATIVE_TOLERANCE) * effect if evaluable else None,
            "control_update_lower": values.get("update_lower", 6400) if values else None,
            "control_update_upper": values.get("update_upper", 6400) if values else None})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--wall-time-minutes", type=float, default=60)
    args = parser.parse_args()
    root = project_dir()
    original = root / "runs/training_length/coefficient_lr_branches_20261010_01"
    holdout = root / "runs/training_length/coefficient_lr_multiplier2_20261010_02"
    assert json.loads((holdout / "status.json").read_text())["complete"]
    output = root / args.output_dir
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    for folder in ("parents", "validation", "branches", "source_snapshot"):
        (output / folder).mkdir()
    started = time.perf_counter()
    device = resolve_device(args.device)
    set_reproducibility(0)
    shared = ("run_coefficient_lr_branches.py", "run_training_length_pilot.py", "stage3_simplified_attention.py",
        "run_training_length_trace.py", "run_matched_initial_control.py", "run_training_length_init.py", "analyze_stage3_mechanism.py")
    assert all(file_digest(root / "src" / name) == file_digest(original / "source_snapshot" / name) for name in shared)
    rule = calibrate_rule(original, holdout)
    write_json(output / "rule_frozen_before_new_training.json", rule)
    rule_hash = file_digest(output / "rule_frozen_before_new_training.json")
    manifest = {"date": "2026-10-10", "arguments": vars(args), "seeds": list(SEEDS), "lengths": list(LENGTHS),
        "rates": {"control": 0.003, "x2": 0.006}, "other_parameter_lr": 0.003,
        "rule_sha256": rule_hash, "source_hashes_match_validated_code": True,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root.parent, text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--short"], cwd=root.parent, text=True),
        "python": platform.python_version(), "torch": torch.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "execution_order": "freeze rule, four prefixes, four controls, freeze eight numeric predictions, four x2 tails",
        "ineligible_parent_policy": "retain both paths as diagnostic; do not score post-fitting prediction or replace seed"}
    write_json(output / "manifest.json", manifest)
    for name in (*shared, "run_coefficient_lr_holdout.py", "analyze_coefficient_lr_branches.py",
                 "analyze_coefficient_lr_holdout.py", Path(__file__).name):
        shutil.copy2(root / "src" / name, output / "source_snapshot" / name)
    shutil.copy2(root / "documents/training_length/COEFFICIENT_LR_NEWSEEDS_SPEC.md", output / "source_snapshot/COEFFICIENT_LR_NEWSEEDS_SPEC.md")
    write_csv(output / "old_seed_calibration.csv", rule["calibration_ratios"])
    original_config = json.loads((root / "runs/training_length/seed_replication_20260929_01/learned_log_n10_s3/config.json").read_text())
    settings = {key: value for key, value in original_config.items() if key in Stage3Config.__dataclass_fields__}
    settings["train_lengths"], settings["eval_lengths"] = tuple(settings["train_lengths"]), tuple(settings["eval_lengths"])
    parents, eligibility, rows, predictions = [], [], [], []
    initial_hashes = {}
    def check_time():
        remaining = args.wall_time_minutes * 60 - (time.perf_counter() - started)
        assert remaining > max((row["wall_seconds"] * 1.2 for row in rows), default=60), "Execution time cap"
    for seed in SEEDS:
        for length in LENGTHS:
            check_time()
            name = f"n{length}_s{seed}"
            directory = output / "parents" / name
            directory.mkdir()
            config = replace(Stage3Config(**settings), seed=seed, device=args.device, train_length=length,
                train_lengths=(length,), max_train_steps=6400, output_dir=directory.relative_to(root).as_posix())
            prefix = replace(config, max_train_steps=1600)
            recorder = TrainingRecorder(prefix, directory, 200)
            trace = BranchTrace(directory, PREFIX_STEPS)
            captured = []
            def observer(model, tokens, labels):
                recorder(model, tokens, labels)
                if tokens is None:
                    trace.record_norms(model, 0)
            def boundary(model, optimizer, train, val, epoch, updates, history):
                if updates == 1600:
                    bundle = capture_parent(config, model, optimizer, train, val, epoch, updates, history, recorder, trace)
                    torch.save(bundle, directory / "parent_state.pt")
                    captured.append(bundle)
            print(f"PARENT {name}", flush=True)
            set_reproducibility(seed)
            model, updates = train_model(prefix, device=device, output_dir=directory,
                training_observer=observer, update_observer=trace, epoch_observer=boundary)
            assert updates == 1600 and len(captured) == 1
            bundle = captured[0]
            if seed in initial_hashes:
                assert initial_hashes[seed] == recorder.initial_digest
            initial_hashes[seed] = recorder.initial_digest
            metrics = initial_metrics(model)
            tokens, _ = fixed_inputs(length, device)
            with torch.no_grad():
                logits = model(tokens)
            eligible = bool((logits[:3] > 0).all() and logits[-1] < 0 and metrics["head_slope"] > 0 and metrics["delta"] > 0)
            e = {"seed": seed, "length": length, "eligible": eligible, "positive_logit": float(logits[0]),
                "negative_logit": float(logits[-1]), "initial_hash": recorder.initial_digest, **metrics,
                "parent_sha256": tree_digest(bundle), "checkpoint_file_sha256": file_digest(directory / "parent_state.pt")}
            eligibility.append(e)
            write_json(directory / "eligibility.json", e)
            check = short_parent_checks(bundle, config, device, output / "validation" / name)
            write_json(output / "validation" / name / "group_verification.json", check)
            parents.append((name, config, bundle, eligible, e))
            print(f"READY {name} eligible={eligible} r={metrics['c_delta']:.6f}", flush=True)
    write_csv(output / "parent_eligibility.csv", eligibility)
    for tag, multiplier in (("control", 1.0), ("x2", 2.0)):
        if tag == "x2":
            assert len(predictions) == 8
            write_csv(output / "predictions_frozen_before_x2.csv", predictions)
            write_json(output / "prediction_freeze.json", {"all_controls_completed": True, "no_x2_tail_started": True,
                "rule_sha256": rule_hash, "numeric_predictions_sha256": file_digest(output / "predictions_frozen_before_x2.csv"),
                "elapsed_seconds": time.perf_counter() - started})
        for name, parent_config, bundle, eligible, e in parents:
            check_time()
            directory = output / "branches" / f"{name}_{tag}"
            directory.mkdir()
            config = replace(parent_config, output_dir=directory.relative_to(root).as_posix())
            recorder = restored_recorder(bundle, config, directory)
            trace = BranchTrace(directory, TAIL_STEPS, 1600, bundle["norm_rows"])
            branch_started = time.perf_counter()
            print(f"BRANCH {len(rows)+1}/8 {name}_{tag}", flush=True)
            model, optimizer, train, val = continue_training(bundle, config, device, directory, multiplier, recorder, trace)
            assert trace.step == 6400 and len(trace.rows) == 232
            assert max(max(trace.max_residual.values()), trace.max_allocation_residual) < 1e-10
            torch.save({"named_optimizer": named_optimizer_state(model, optimizer), "optimizer": cpu_tree(optimizer.state_dict()),
                "rng": capture_rng(), "train_loaders": loader_state(train), "val_loaders": loader_state(val),
                "capture_boundary": "after final validation, before final evaluation"}, directory / "final_continuation_state.pt")
            row = finish_run(config, directory, device, model, 6400, recorder, branch_started)
            first = torch.load(directory / "first_update.pt", map_location="cpu", weights_only=False)
            if tag == "x2":
                control = output / "branches" / f"{name}_control"
                old_first = torch.load(control / "first_update.pt", map_location="cpu", weights_only=False)
                assert tree_digest(first["before"]) == tree_digest(old_first["before"])
                assert tree_digest(first["after_moments"]) == tree_digest(old_first["after_moments"])
                assert all(torch.equal(first["after_weights"][key], old_first["after_weights"][key])
                    for key in first["after_weights"] if key != "alpha_log_scale_unconstrained")
                assert read_rows(directory / "batch_progress.csv") == read_rows(control / "batch_progress.csv")
                assert read_rows(directory / "train_history.csv")[:50] == read_rows(control / "train_history.csv")[:50]
                write_json(directory / "sibling_verification.json", {"initial_states_gradients_moments_exact": True,
                    "first_other_parameter_updates_exact": True, "all_tail_batches_exact": True, "prefix_history_exact": True})
            else:
                path = load_trajectory(directory, int(config.train_lengths[0]))
                predictions.extend(predict_from_control(rule, config.seed, config.train_lengths[0], path[0]["c_delta"], path, eligible))
            assert file_digest(output / "parents" / name / "parent_state.pt") == e["checkpoint_file_sha256"]
            assert file_digest(output / "rule_frozen_before_new_training.json") == rule_hash
            write_json(directory / "branch_verification.json", {"parent_sha256": tree_digest(bundle),
                "eligible_parent": eligible, "tail_updates": 4800, "traced_updates": 232,
                "max_decomposition_residual": max(max(trace.max_residual.values()), trace.max_allocation_residual),
                "total_allocation": trace.total_allocation, "lr_multiplier": multiplier})
            row.update(tag=tag, multiplier=multiplier, parent=name, eligible_parent=eligible)
            rows.append(row)
            write_csv(output / "summary.csv", rows)
            write_json(output / "status.json", {"complete": len(rows) == 8, "completed_parents": 4,
                "eligible_parents": sum(e["eligible"] for e in eligibility), "completed_branches": len(rows),
                "planned_branches": 8, "wall_seconds": time.perf_counter() - started,
                "scheduled_updates": 44800, "group_check_updates": 256, "seeds_replaced": False})
            print(f"DONE {name}_{tag} r={row['c_delta']:.6f} seconds={row['wall_seconds']:.2f}", flush=True)
    for source, digest in rule["source_sha256"].items():
        assert file_digest(Path(source)) == digest
    assert file_digest(output / "predictions_frozen_before_x2.csv") == json.loads((output / "prediction_freeze.json").read_text())["numeric_predictions_sha256"]
    print("COMPLETE 8 independent-seed comparison paths", flush=True)


if __name__ == "__main__":
    main()
