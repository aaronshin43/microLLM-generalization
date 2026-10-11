"""Execute the frozen multiplier-2 predictions from existing full parent states."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import torch

from run_coefficient_lr_branches import (
    BranchTrace, TAIL_STEPS, capture_rng, continue_training, cpu_tree,
    first_epoch_batch, loader_state, named_optimizer_state, read_rows,
    restore_parent, restored_recorder, split_coefficient_group, tree_digest,
)
from run_training_length_pilot import finish_run
from run_training_length_trace import trace_state
from stage3_simplified_attention import Stage3Config, project_dir, resolve_device, set_reproducibility, write_csv, write_json


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_update_proposal(bundle, device):
    model, optimizer, train, _ = restore_parent(bundle, device)
    split_coefficient_group(model, optimizer, 2.0)
    tokens, labels = first_epoch_batch(bundle, train)[:2]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.binary_cross_entropy_with_logits(model(tokens.to(device)), labels.to(device)).backward()
    optimizer.step()
    return {"weights": cpu_tree(model.state_dict()), "moments": named_optimizer_state(model, optimizer),
            "state": trace_state(model), "kind": "Discarded first-update implementation check"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--wall-time-minutes", type=float, default=60)
    args = parser.parse_args()
    root = project_dir()
    source = root / args.parent_run_dir
    output = root / args.output_dir
    if output.exists():
        raise FileExistsError(output)
    original_status = json.loads((source / "status.json").read_text())
    assert original_status["complete"] and original_status["completed_branches"] == 18
    prediction_source = source / "analysis/held_out_multiplier2_predictions.csv"
    predictions = read_rows(prediction_source)
    assert len(predictions) == 12
    assert {(int(row["seed"]), int(row["length"]), row["comparison"]) for row in predictions} == {
        (seed, length, comparison) for seed in (3, 4, 6) for length in (10, 100)
        for comparison in ("update6400", "common100_bce0.01")}
    assert all(float(row["coefficient_multiplier"]) == 2 and float(row["coefficient_lr"]) == 0.006
               and float(row["tolerance_relative_effect"]) == 0.2 and row["executed"] == "False" for row in predictions)
    # Reuse only the exact continuation code validated by the original experiment.
    shared = ("run_coefficient_lr_branches.py", "run_training_length_pilot.py",
              "stage3_simplified_attention.py", "run_training_length_trace.py",
              "run_matched_initial_control.py", "run_training_length_init.py", "analyze_stage3_mechanism.py")
    assert all(file_digest(root / "src" / name) == file_digest(source / "source_snapshot" / name) for name in shared)
    device = resolve_device(args.device)
    # Original parent training enabled deterministic kernels; RNG restoration alone does not restore this flag.
    set_reproducibility(0)
    original_manifest = json.loads((source / "manifest.json").read_text())
    assert str(torch.__version__) == original_manifest["torch"]
    assert str(device) == original_manifest["device"]
    if device.type == "cuda":
        assert torch.cuda.get_device_name(device) == original_manifest["gpu"]
    output.mkdir(parents=True)
    for name in ("frozen_inputs", "branches", "source_snapshot"):
        (output / name).mkdir()
    shutil.copy2(prediction_source, output / "frozen_inputs/predictions.csv")
    prediction_hash = file_digest(prediction_source)
    started = time.perf_counter()
    parents = []
    for seed in (3, 4, 6):
        for length in (10, 100):
            name = f"n{length}_s{seed}"
            parent_file = source / "parents" / name / "parent_state.pt"
            verification = json.loads((parent_file.parent / "parent_verification.json").read_text())
            assert file_digest(parent_file) == verification["checkpoint_file_sha256"]
            shutil.copy2(parent_file, output / "frozen_inputs" / f"{name}_parent.pt")
            bundle = torch.load(parent_file, map_location="cpu", weights_only=False)
            assert tree_digest(bundle) == verification["parent_state_sha256"]
            assert bundle["optimizer_updates"] == 1600 and bundle["next_epoch"] == 51
            config = replace(Stage3Config(**bundle["config"]), device=args.device,
                output_dir=(output / "branches" / f"{name}_x2").relative_to(root).as_posix())
            assert config.max_train_steps == 6400 and config.learning_rate == 0.003
            proposal = first_update_proposal(bundle, device)
            torch.save(proposal, output / "frozen_inputs" / f"{name}_first_update.pt")
            parents.append((name, config, bundle, proposal, parent_file, verification))
    manifest = {"date": "2026-10-10", "launch_arguments": vars(args), "parent_run": str(source),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root.parent, text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--short"], cwd=root.parent, text=True),
        "python": platform.python_version(), "torch": torch.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "prediction_source_sha256": prediction_hash, "six_parent_forecasts_saved_before_tails": True,
        "coefficient_lr": 0.006, "other_lr": 0.003, "shared_source_hashes_equal": True,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "scheduled_updates": 28800, "disposable_proposal_updates": 6}
    write_json(output / "manifest.json", manifest)
    for name in (*shared, Path(__file__).name):
        shutil.copy2(root / "src" / name, output / "source_snapshot" / name)
    shutil.copy2(root / "documents/training_length/COEFFICIENT_LR_MULTIPLIER2_SPEC.md", output / "source_snapshot/COEFFICIENT_LR_MULTIPLIER2_SPEC.md")
    write_json(output / "predictions_frozen.json", {"before_training": True,
        "prediction_source_sha256": prediction_hash, "copied_file_sha256": file_digest(output / "frozen_inputs/predictions.csv"),
        "parent_state_sha256": {name: verification["parent_state_sha256"] for name, _, _, _, _, verification in parents},
        "model_and_tolerance_unchanged": True})
    rows = []
    for name, config, bundle, proposal, parent_file, parent_verification in parents:
        remaining = args.wall_time_minutes * 60 - (time.perf_counter() - started)
        assert remaining > max((row["wall_seconds"] * 1.2 for row in rows), default=60), "Execution time cap"
        directory = root / config.output_dir
        directory.mkdir()
        recorder = restored_recorder(bundle, config, directory)
        trace = BranchTrace(directory, TAIL_STEPS, 1600, bundle["norm_rows"])
        branch_started = time.perf_counter()
        print(f"START {len(rows)+1}/6 {name}_x2", flush=True)
        model, optimizer, train, val = continue_training(bundle, config, device, directory, 2.0, recorder, trace)
        assert trace.step == 6400 and len(trace.rows) == 232
        assert max(max(trace.max_residual.values()), trace.max_allocation_residual) < 1e-10
        torch.save({"named_optimizer": named_optimizer_state(model, optimizer),
            "optimizer": cpu_tree(optimizer.state_dict()), "rng": capture_rng(),
            "train_loaders": loader_state(train), "val_loaders": loader_state(val),
            "capture_boundary": "after final epoch validation, before final evaluation"}, directory / "final_continuation_state.pt")
        row = finish_run(config, directory, device, model, 6400, recorder, branch_started)
        first = torch.load(directory / "first_update.pt", map_location="cpu", weights_only=False)
        control = source / "branches" / f"{name}_control"
        old_first = torch.load(control / "first_update.pt", map_location="cpu", weights_only=False)
        assert tree_digest(first["before"]) == tree_digest(old_first["before"])
        assert tree_digest(first["after_moments"]) == tree_digest(old_first["after_moments"])
        assert tree_digest(first["after_weights"]) == tree_digest(proposal["weights"])
        for key in first["after_weights"]:
            if key != "alpha_log_scale_unconstrained":
                assert torch.equal(first["after_weights"][key], old_first["after_weights"][key])
        assert read_rows(directory / "batch_progress.csv") == read_rows(control / "batch_progress.csv")
        assert file_digest(parent_file) == parent_verification["checkpoint_file_sha256"]
        assert file_digest(prediction_source) == prediction_hash
        history = read_rows(directory / "train_history.csv")
        assert history[:50] == read_rows(control / "train_history.csv")[:50]
        verification = {"parent_state_sha256": tree_digest(bundle), "initial_state_gradient_and_moments_exact": True,
            "first_other_parameter_updates_exact": True, "first_update_proposal_exact": True,
            "all_tail_batches_match_control": True, "parent_and_prediction_files_unchanged": True,
            "prefix_history_exact": True, "tail_updates": 4800, "traced_updates": 232,
            "max_decomposition_residual": max(max(trace.max_residual.values()), trace.max_allocation_residual),
            "total_allocation": trace.total_allocation, "coefficient_lr": 0.006, "other_lr": 0.003}
        write_json(directory / "branch_verification.json", verification)
        row.update(parent=name, multiplier=2.0, tag="x2", traced_updates=232,
                   tail_batch_sha256=trace.batch_digest.hexdigest())
        rows.append(row)
        write_csv(output / "summary.csv", rows)
        write_json(output / "status.json", {"complete": len(rows) == 6, "completed_branches": len(rows),
            "planned_branches": 6, "wall_seconds": time.perf_counter() - started,
            "scheduled_updates": 28800, "replayed_prefix_updates": 0, "disposable_proposal_updates": 6})
        print(f"DONE {name}_x2 seconds={row['wall_seconds']:.2f} r={row['c_delta']:.6f}", flush=True)
    assert file_digest(prediction_source) == file_digest(output / "frozen_inputs/predictions.csv") == prediction_hash
    print("COMPLETE 6 held-out branches", flush=True)


if __name__ == "__main__":
    main()
