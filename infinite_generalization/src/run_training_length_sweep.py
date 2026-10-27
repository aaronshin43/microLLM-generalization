"""Run nine baseline lengths with passive, complete scalar AdamW histories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import torch

from run_coefficient_lr_branches import (capture_parent, capture_rng, cpu_tree, loader_state,
    named_optimizer_state, read_rows, tree_digest)
from run_coefficient_lr_holdout import file_digest
from run_training_length_pilot import (TrainingRecorder, attention_gap, finish_run, mechanism,
    sigmoid, state_digest)
from run_training_length_trace import trace_state
from stage3_simplified_attention import (Stage3Config, SimplifiedLastQueryAttentionClassifier,
    evaluate_length, project_dir, resolve_device, set_reproducibility, train_model, write_csv, write_json)

SEEDS = (3, 4, 6)
LENGTHS = (10, 30, 100, 300, 1000)
NEW_LENGTHS = (30, 300, 1000)
EVAL_LENGTHS = (*LENGTHS, 10000)
COEFFICIENT = "alpha_log_scale_unconstrained"


def config_from_record(record):
    settings = {key: value for key, value in record.items() if key in Stage3Config.__dataclass_fields__}
    for key in ("train_lengths", "eval_lengths"):
        settings[key] = tuple(settings[key])
    return Stage3Config(**settings)


def settings_difference(a, b):
    return {key: [asdict(a)[key], asdict(b)[key]] for key in asdict(a)
        if asdict(a)[key] != asdict(b)[key]}


def audit_references(root, output):
    baseline = root / "runs/training_length/seed_replication_20260929_01"
    branch = root / "runs/training_length/coefficient_lr_branches_20261010_01"
    inventory, initial_hashes, template = [], {}, None
    for seed in SEEDS:
        for length in (10, 100):
            name = f"n{length}_s{seed}"
            old = baseline / f"learned_log_{name}"
            control = branch / "branches" / f"{name}_control"
            parent = branch / "parents" / name
            record = json.loads((old / "config.json").read_text())
            config = config_from_record(record)
            if template is None:
                template = config
            differences = settings_difference(template, config)
            assert set(differences) <= {"seed", "train_length", "train_lengths", "output_dir"}, differences
            initial_hashes.setdefault(seed, record["initial_state_sha256"])
            assert initial_hashes[seed] == record["initial_state_sha256"]
            a = torch.load(old / "model.pt", map_location="cpu", weights_only=False)
            b = torch.load(control / "model.pt", map_location="cpu", weights_only=False)
            assert a["optimizer_updates"] == b["optimizer_updates"] == 6400
            assert tree_digest(a["state_dict"]) == tree_digest(b["state_dict"])
            for filename in ("training_diagnostics.csv", "train_history.csv"):
                assert read_rows(old / filename) == read_rows(control / filename)
            pstate = torch.load(parent / "parent_state.pt", map_location="cpu", weights_only=False)
            final_state = torch.load(control / "final_continuation_state.pt", map_location="cpu", weights_only=False)
            pcheck = json.loads((parent / "parent_verification.json").read_text())
            assert file_digest(parent / "parent_state.pt") == pcheck["checkpoint_file_sha256"]
            assert pstate["recorder"]["initial_digest"] == record["initial_state_sha256"]
            group = pstate["optimizer"]["param_groups"][0]
            assert group["lr"] == 0.003 and group["weight_decay"] == 0
            assert tuple(group["betas"]) == (0.9, 0.999) and group["eps"] == 1e-8
            assert set(final_state["named_optimizer"]) == set(pstate["named_optimizer"])
            files = [old / f for f in ("config.json", "model.pt", "training_diagnostics.csv",
                "train_history.csv", "metrics_by_length.csv", "exposure.json")]
            files += [parent / "parent_state.pt", parent / "parent_verification.json",
                parent / "update_trace.csv", control / "model.pt", control / "update_trace.csv",
                control / "training_diagnostics.csv", control / "train_history.csv",
                control / "final_continuation_state.pt", control / "baseline_verification.json"]
            parent_steps = [int(row["optimizer_updates"]) for row in read_rows(parent / "update_trace.csv")]
            tail_steps = [int(row["optimizer_updates"]) for row in read_rows(control / "update_trace.csv")]
            early = root / f"runs/training_length/update_trace_20260929_01/learned_log_{name}_baseline/update_trace.csv"
            early_steps = []
            if early.exists():
                files.append(early)
                early_steps = [int(row["optimizer_updates"]) for row in read_rows(early)]
            inventory.append({"seed": seed, "train_length": length,
                "baseline": str(old), "control": str(control), "parent": str(parent),
                "initial_state_sha256": record["initial_state_sha256"],
                "model_final_sha256": tree_digest(a["state_dict"]),
                "baseline_control_weights_history_diagnostics_exact": True,
                "optimizer_options": {k: v for k, v in group.items() if k != "params"},
                "optimizer_full_states": [1600, 6400], "complete_coefficient_history": False,
                "detailed_moment_steps": parent_steps + tail_steps,
                "additional_gradient_only_steps": early_steps,
                "missing_history_policy": "no interpolation or baseline replays; second-moment reconstruction unavailable",
                "source_sha256": {str(path): file_digest(path) for path in files}})
    write_json(output / "reference_inventory.json", {"cells": inventory})
    return template, initial_hashes, inventory


def coefficient_state(model, optimizer):
    parameter = model.alpha_log_scale_unconstrained
    state = optimizer.state.get(parameter, {})
    group = next(group for group in optimizer.param_groups if any(p is parameter for p in group["params"]))
    step = int(state.get("step", 0))
    m, v = float(state.get("exp_avg", 0)), float(state.get("exp_avg_sq", 0))
    beta1, beta2 = group["betas"]
    return {"step": step, "m": m, "v": v,
        "m_hat": m / (1 - beta1 ** step) if step else 0.0,
        "v_hat": v / (1 - beta2 ** step) if step else 0.0,
        "lr": group["lr"], "beta1": beta1, "beta2": beta2, "eps": group["eps"]}


class CoefficientTrace:
    """Observe scalar updates without modifying gradients, moments, RNG or modes."""

    def __init__(self, output, max_steps, deadline=None):
        self.output, self.max_steps, self.deadline = output, max_steps, deadline
        self.step = 0
        self.rows, self.pending, self.blocks, self.norm_rows = [], [], [], []
        self.batch_digest = hashlib.sha256()
        self.batch_rows = []

    def before_step(self, model, tokens, labels, loss, optimizer):
        if self.deadline is not None and self.step % 200 == 0 and time.perf_counter() >= self.deadline:
            raise TimeoutError("Sweep wall-time ceiling reached; partial artifacts retained")
        self.step += 1
        self.before = trace_state(model), coefficient_state(model, optimizer)
        self.gradient = float(model.alpha_log_scale_unconstrained.grad)
        self.batch = (tokens.shape[0], tokens.shape[1], int(labels.sum().item()))
        self.batch_digest.update(tree_digest((self.step, tokens, labels)).encode())

    def after_step(self, model, tokens, labels, loss, optimizer):
        a, old = self.before
        b, new = trace_state(model), coefficient_state(model, optimizer)
        dc, dd = b["c"] - a["c"], b["delta"] - a["delta"]
        dr = b["c"] * b["delta"] - a["c"] * a["delta"]
        c_part, score_part = (a["delta"] + b["delta"]) * dc / 2, (a["c"] + b["c"]) * dd / 2
        direction = -new["m_hat"] / (math.sqrt(new["v_hat"]) + new["eps"])
        row = {"optimizer_updates": self.step, "batch_size": self.batch[0], "batch_length": self.batch[1],
            "positives": self.batch[2], "negatives": self.batch[0] - self.batch[2],
            "u_before": a["u"], "u_after": b["u"], "u_gradient": self.gradient,
            "u_update": b["u"] - a["u"], "c_before": a["c"], "c_after": b["c"], "c_update": dc,
            "delta_before": a["delta"], "delta_after": b["delta"], "delta_update": dd,
            "r_before": a["c"] * a["delta"], "r_after": b["c"] * b["delta"],
            "r_update": dr, "r_from_coefficient": c_part, "r_from_score": score_part,
            "r_residual": dr - c_part - score_part, "normalized_direction": direction,
            "actual_u_update_over_lr": (b["u"] - a["u"]) / new["lr"],
            "learning_rate": new["lr"], "beta1": new["beta1"], "beta2": new["beta2"], "epsilon": new["eps"]}
        for suffix, state in (("before", old), ("after", new)):
            row.update({f"{key}_{suffix}": state[key] for key in ("step", "m", "v", "m_hat", "v_hat")})
        self.rows.append(row)
        self.pending.append(row)
        if self.step % 200 == 0 or self.step == self.max_steps:
            path = self.output / "coefficient_trace.csv"
            with path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                if self.step == len(self.pending):
                    writer.writeheader()
                writer.writerows(self.pending)
            block = {"update_lower": self.pending[0]["optimizer_updates"], "update_upper": self.step,
                "u_update": sum(r["u_update"] for r in self.pending), "c_update": sum(r["c_update"] for r in self.pending),
                "r_update": sum(r["r_update"] for r in self.pending),
                "r_from_coefficient": sum(r["r_from_coefficient"] for r in self.pending),
                "r_from_score": sum(r["r_from_score"] for r in self.pending),
                "max_abs_residual": max(abs(r["r_residual"]) for r in self.pending),
                "mean_abs_gradient": sum(abs(r["u_gradient"]) for r in self.pending) / len(self.pending),
                "mean_normalized_direction": sum(r["normalized_direction"] for r in self.pending) / len(self.pending),
                "m_after": new["m"], "v_after": new["v"]}
            self.blocks.append(block)
            self.batch_rows.append({"optimizer_updates": self.step, "rolling_batch_sha256": self.batch_digest.hexdigest()})
            write_csv(self.output / "allocation_by_200.csv", self.blocks)
            write_csv(self.output / "batch_progress.csv", self.batch_rows)
            self.pending = []


def passive_recording_check(config, device, output):
    output.mkdir()
    target = replace(config, max_train_steps=64)
    results = []
    for observed in (False, True):
        directory = output / ("observed" if observed else "unobserved")
        directory.mkdir()
        recorder = TrainingRecorder(target, directory, 32, diagnostic_lengths=EVAL_LENGTHS) if observed else None
        trace = CoefficientTrace(directory, 64) if observed else None
        snapshots = []
        def boundary(model, optimizer, train, val, epoch, updates, history):
            snapshots.append({"model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
                "named_optimizer": named_optimizer_state(model, optimizer), "rng": capture_rng(),
                "train": loader_state(train), "val": loader_state(val), "history": list(history)})
        set_reproducibility(config.seed)
        model, updates = train_model(target, device=device, output_dir=directory,
            training_observer=recorder, update_observer=trace, epoch_observer=boundary)
        results.append({"snapshots": snapshots, "history": read_rows(directory / "train_history.csv"),
            "updates": updates, "final_model": cpu_tree(model.state_dict())})
    assert tree_digest(results[0]) == tree_digest(results[1]), "Recording changed training"
    return {"length": config.train_length, "seed": config.seed, "steps_per_path": 64,
        "discarded_updates": 128, "weights_optimizer_rng_loaders_history_exact": True}


def evaluate_saved_baseline(reference, config, device, output):
    output.mkdir()
    saved_rng = capture_rng()
    model = SimplifiedLastQueryAttentionClassifier(d_head=config.d_head, alpha_mode=config.alpha_mode,
        alpha_log_scale_init=config.alpha_log_scale_init).to(device)
    old = torch.load(Path(reference["baseline"]) / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(old["state_dict"])
    values = mechanism(model)
    historical = {int(row["length"]): row for row in read_rows(Path(reference["baseline"]) / "metrics_by_length.csv")}
    rows = []
    for length in EVAL_LENGTHS:
        if length in historical:
            row = {**historical[length], "provenance": "historical_explicit_evaluation"}
        else:
            row, _, _, _ = evaluate_length(model, length=length, examples=config.test_examples,
                eval_chunk_examples=config.eval_chunk_examples, eval_sampling_mode=config.eval_sampling_mode,
                batch_size=config.eval_batch_size, seed=config.seed + 10000 + length,
                target_position_mode=config.target_position_mode, target_token_count=1, non_target_token_count=1,
                non_target_sampling=config.non_target_sampling, device=device)
            row["provenance"] = "saved_weight_evaluation_no_training"
        gap, _ = attention_gap(config.alpha_mode, values, math.log(length))
        assert math.isclose(float(row["mean_empirical_target_attention"]), sigmoid(gap), rel_tol=3e-5, abs_tol=3e-6)
        rows.append(row)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(output / "metrics_by_length.csv", [{key: row.get(key) for key in keys} for row in rows])
    from run_coefficient_lr_branches import restore_rng
    restore_rng(saved_rng)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--wall-time-minutes", default=60, type=float)
    args = parser.parse_args()
    root = project_dir()
    output = root / args.output_dir
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    for folder in ("source_snapshot", "validation", "new_runs", "reused_evaluations"):
        (output / folder).mkdir()
    started = time.perf_counter()
    deadline = started + 60 * args.wall_time_minutes
    device = resolve_device(args.device)
    set_reproducibility(0)
    template, initial_hashes, references = audit_references(root, output)
    order = [(length, seed) for length in NEW_LENGTHS for seed in SEEDS]
    configurations = [replace(template, seed=seed, train_length=length, train_lengths=(length,),
        eval_lengths=EVAL_LENGTHS, device=args.device,
        output_dir=(output / "new_runs" / f"n{length}_s{seed}").relative_to(root).as_posix()) for length, seed in order]
    for config in configurations:
        assert set(settings_difference(template, config)) <= {"seed", "train_length", "train_lengths", "eval_lengths", "device", "output_dir"}
    sources = (Path(__file__).name, "run_training_length_pilot.py", "stage3_simplified_attention.py",
        "run_training_length_trace.py", "run_coefficient_lr_branches.py", "run_coefficient_lr_holdout.py",
        "analyze_stage3_mechanism.py")
    for name in sources:
        shutil.copy2(root / "src" / name, output / "source_snapshot" / name)
    for name in ("LENGTH_SWEEP_PLAN.md", "LENGTH_SWEEP_SPEC.md"):
        shutil.copy2(root / "documents/training_length" / name, output / "source_snapshot" / name)
    manifest = {"date": "2026-10-24", "arguments": vars(args), "lengths": LENGTHS, "seeds": SEEDS,
        "new_order": order, "new_configurations": [asdict(c) for c in configurations],
        "source_sha256": {name: file_digest(output / "source_snapshot" / name) for name in sources},
        "reference_inventory_sha256": file_digest(output / "reference_inventory.json"),
        "initial_hashes": initial_hashes, "common100_bce_targets": [0.02, 0.01, 0.005],
        "primary_loss_target": 0.01, "total_new_scientific_updates": 57600,
        "expected_sequence_exposures": 3686400, "expected_token_exposures": 1634304000,
        "discarded_check_updates": 384, "python": platform.python_version(), "torch": torch.__version__,
        "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dtype": str(torch.get_default_dtype()), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root.parent, text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--short"], cwd=root.parent, text=True),
        "no_replays_or_interventions": True}
    write_json(output / "manifest.json", manifest)
    rows, checks = [], []
    status = {"complete": False, "planned_new_runs": 9, "completed_new_runs": 0, "reused_cells": 6,
        "scientific_updates": 0, "discarded_updates": 0, "phase": "passive_recording_checks"}
    try:
        for length in NEW_LENGTHS:
            config = next(c for c in configurations if c.seed == 3 and c.train_length == length)
            check = passive_recording_check(config, device, output / "validation" / f"n{length}_s3")
            checks.append(check)
            status["discarded_updates"] += check["discarded_updates"]
            write_json(output / "passive_recording_checks.json", {"checks": checks})
            print(f"CHECK length={length} exact=True", flush=True)
        for index, config in enumerate(configurations):
            if time.perf_counter() >= deadline:
                raise TimeoutError("Sweep wall-time ceiling reached before next run")
            directory = root / config.output_dir
            directory.mkdir()
            recorder = TrainingRecorder(config, directory, 200, diagnostic_lengths=EVAL_LENGTHS)
            trace = CoefficientTrace(directory, 6400, deadline)
            checkpoint_hashes = {}
            def boundary(model, optimizer, train, val, epoch, updates, history):
                if updates in (1600, 6400):
                    bundle = capture_parent(config, model, optimizer, train, val, epoch, updates, history, recorder, trace)
                    path = directory / f"state_step{updates}.pt"
                    torch.save(bundle, path)
                    checkpoint_hashes[str(updates)] = file_digest(path)
            run_started = time.perf_counter()
            status.update(phase="scientific_training", active_run=directory.name)
            write_json(output / "status.json", status)
            print(f"RUN {index+1}/9 {directory.name}", flush=True)
            write_json(directory / "config.json", asdict(config))
            set_reproducibility(config.seed)
            model, updates = train_model(config, device=device, output_dir=directory,
                training_observer=recorder, update_observer=trace, epoch_observer=boundary)
            assert recorder.initial_digest == initial_hashes[config.seed], "Paired initialization mismatch"
            assert updates == trace.step == 6400 and len(trace.rows) == 6400 and len(trace.blocks) == 32
            assert max(abs(row["r_residual"]) for row in trace.rows) < 1e-10
            row = finish_run(config, directory, device, model, updates, recorder, run_started, explicit_lengths=EVAL_LENGTHS)
            assert len(recorder.rows) == 33 * 6
            assert row["total_sequences"] == 409600 and row["total_tokens"] == 409600 * config.train_length
            assert recorder.exposure[str(config.train_length)]["positives"] == 204800
            write_json(directory / "verification.json", {"initial_hash_matches_reused_seed": True,
                "complete_coefficient_history": True, "coefficient_rows": len(trace.rows),
                "checkpoint_sha256": checkpoint_hashes,
                "max_allocation_residual": max(abs(r["r_residual"]) for r in trace.rows),
                "updates": updates, "recorder_rows": len(recorder.rows)})
            rows.append(row)
            write_csv(output / "summary.csv", rows)
            status.update(completed_new_runs=len(rows), scientific_updates=6400 * len(rows),
                wall_seconds=time.perf_counter() - started)
            write_json(output / "status.json", status)
            print(f"DONE {directory.name} c={row['c']:.6f} delta={row['delta']:.6f} r={row['c_delta']:.6f} seconds={row['wall_seconds']:.1f}", flush=True)
        for reference in references:
            old_config = config_from_record(json.loads((Path(reference["baseline"]) / "config.json").read_text()))
            evaluate_saved_baseline(reference, old_config, device,
                output / "reused_evaluations" / f"n{old_config.train_length}_s{old_config.seed}")
        for reference in references:
            for path, digest in reference["source_sha256"].items():
                assert file_digest(Path(path)) == digest, "Historical file changed"
        status.update(complete=True, phase="complete", active_run=None, wall_seconds=time.perf_counter() - started,
            sequence_exposures=sum(r["total_sequences"] for r in rows),
            token_exposures=sum(r["total_tokens"] for r in rows), historical_sources_unchanged=True)
        assert status["sequence_exposures"] == 3686400 and status["token_exposures"] == 1634304000
        write_json(output / "status.json", status)
        print("COMPLETE nine new runs and six reused evaluation cells", flush=True)
    except Exception as error:
        status.update(phase="incomplete", error=f"{type(error).__name__}: {error}", wall_seconds=time.perf_counter() - started)
        write_json(output / "status.json", status)
        raise


if __name__ == "__main__":
    main()
