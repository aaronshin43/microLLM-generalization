"""Branch successful baselines while preserving AdamW and data progress."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import random
import shutil
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from run_matched_initial_control import MatchedUpdateTrace, fixed_inputs, initial_metrics
from run_training_length_pilot import TrainingRecorder, finish_run, state_digest
from run_training_length_trace import trace_state, verify_replay
from stage3_simplified_attention import (
    Stage3Config, SimplifiedLastQueryAttentionClassifier, make_multilength_loaders,
    project_dir, resolve_device, run_loaders_once, set_reproducibility,
    train_model, write_csv, write_json,
)

PREFIX_STEPS = set(range(1590, 1601))
TAIL_STEPS = set(range(1601, 1801)) | set(range(3190, 3211)) | set(range(6390, 6401))
MULTIPLIERS = (("slow", 1 / 3), ("control", 1.0), ("fast", 3.0))


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def tree_digest(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda key: str(key)):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(str(type(item)).encode())
            for part in item:
                visit(part)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def loader_state(loaders):
    return [{"dataset_sha256": tree_digest(loader.dataset.tensors),
             "batch_size": loader.batch_size, "drop_last": loader.drop_last,
             "num_workers": loader.num_workers, "sampler": type(loader.sampler).__name__,
             "generator": loader.generator.get_state().clone() if loader.generator is not None else None}
            for loader in loaders]


def restore_loaders(loaders, states):
    assert len(loaders) == len(states)
    current = loader_state(loaders)
    for loader, observed, saved in zip(loaders, current, states):
        assert all(observed[key] == saved[key] for key in observed if key != "generator")
        if saved["generator"] is not None:
            loader.generator.set_state(saved["generator"])


def make_loaders(config):
    common = dict(lengths=config.train_lengths, target_position_mode=config.target_position_mode,
                  target_token_count=config.target_token_count, non_target_token_count=config.non_target_token_count,
                  non_target_sampling=config.non_target_sampling)
    train = make_multilength_loaders(**common, examples_per_length=config.train_examples,
                                    batch_size=config.batch_size, shuffle=True, seed=config.seed + 1)
    val = make_multilength_loaders(**common, examples_per_length=config.val_examples,
                                  batch_size=config.eval_batch_size, shuffle=False, seed=config.seed + 2)
    return train, val


def named_optimizer_state(model, optimizer):
    return {name: cpu_tree(optimizer.state[parameter]) for name, parameter in model.named_parameters()}


def capture_parent(config, model, optimizer, train, val, epoch, updates, history, recorder, trace):
    before = capture_rng()
    bundle = {"config": asdict(config), "model": cpu_tree(model.state_dict()),
              "optimizer": cpu_tree(optimizer.state_dict()),
              "named_optimizer": named_optimizer_state(model, optimizer),
              "parameter_names": [name for name, _ in model.named_parameters()],
              "model_training": model.training, "epoch": epoch, "optimizer_updates": updates,
              "next_epoch": epoch + 1, "batch_cursor": 0, "validation_complete": True,
              "history": copy.deepcopy(history), "train_loaders": loader_state(train),
              "val_loaders": loader_state(val), "rng": before,
              "recorder": {"updates": recorder.updates, "exposure": copy.deepcopy(recorder.exposure),
                           "initial_digest": recorder.initial_digest, "rows": copy.deepcopy(recorder.rows)},
              "norm_rows": copy.deepcopy(trace.norm_rows), "scheduler": None, "scaler": None}
    assert tree_digest(before) == tree_digest(capture_rng())
    return bundle


def restore_parent(bundle, device):
    config = Stage3Config(**bundle["config"])
    train, val = make_loaders(config)
    model = SimplifiedLastQueryAttentionClassifier(d_head=config.d_head, alpha_mode=config.alpha_mode,
        alpha_log_scale_init=config.alpha_log_scale_init, target_token_count=config.target_token_count,
        non_target_token_count=config.non_target_token_count).to(device)
    model.load_state_dict(bundle["model"])
    model.train(bundle["model_training"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer.load_state_dict(copy.deepcopy(bundle["optimizer"]))
    assert [name for name, _ in model.named_parameters()] == bundle["parameter_names"]
    assert tree_digest(model.state_dict()) == tree_digest(bundle["model"])
    assert tree_digest(named_optimizer_state(model, optimizer)) == tree_digest(bundle["named_optimizer"])
    restore_loaders(train, bundle["train_loaders"])
    restore_loaders(val, bundle["val_loaders"])
    restore_rng(bundle["rng"])
    return model, optimizer, train, val


def split_coefficient_group(model, optimizer, multiplier):
    assert len(optimizer.param_groups) == 1
    scalar = model.alpha_log_scale_unconstrained
    states_before = tree_digest(named_optimizer_state(model, optimizer))
    group = optimizer.param_groups[0]
    options = {key: copy.deepcopy(value) for key, value in group.items() if key != "params"}
    original_lr = options["lr"]
    group["params"] = [parameter for parameter in group["params"] if parameter is not scalar]
    optimizer.add_param_group({**options, "params": [scalar], "lr": original_lr * multiplier})
    assert tree_digest(named_optimizer_state(model, optimizer)) == states_before
    assert len(group["params"]) + 1 == len(list(model.parameters()))


def restored_recorder(bundle, config, output):
    recorder = TrainingRecorder(config, output, 200)
    for key, value in bundle["recorder"].items():
        setattr(recorder, key, copy.deepcopy(value))
    write_csv(output / "training_diagnostics.csv", recorder.rows)
    return recorder


def continue_training(bundle, config, device, output, multiplier, recorder, trace):
    model, optimizer, train, val = restore_parent(bundle, device)
    if multiplier is not None:
        split_coefficient_group(model, optimizer, multiplier)
    assert tree_digest(capture_rng()) == tree_digest(bundle["rng"])
    history = copy.deepcopy(bundle["history"])
    updates, epoch = bundle["optimizer_updates"], bundle["epoch"]
    criterion = torch.nn.BCEWithLogitsLoss()
    while updates < config.max_train_steps:
        epoch += 1
        train_loss, metrics, added = run_loaders_once(model, train, criterion=criterion, device=device,
            optimizer=optimizer, max_steps=config.max_train_steps - updates,
            batch_shuffle_seed=config.seed + 20000 + epoch,
            training_observer=recorder, update_observer=trace)
        updates += added
        val_loss, val_metrics, _ = run_loaders_once(model, val, criterion=criterion, device=device)
        history.append({"epoch": epoch, "optimizer_updates": updates, "updates_this_epoch": added,
            "train_loss": train_loss, "train_accuracy": metrics["accuracy"],
            "train_positive_accuracy": metrics["positive_accuracy"], "train_negative_accuracy": metrics["negative_accuracy"],
            "val_loss": val_loss, "val_accuracy": val_metrics["accuracy"],
            "val_positive_accuracy": val_metrics["positive_accuracy"], "val_negative_accuracy": val_metrics["negative_accuracy"]})
    write_csv(output / "train_history.csv", history)
    return model, optimizer, train, val


def mini_state(model):
    q = model.query_projection.weight.detach().cpu().double()[:, 1]
    key = model.key_projection.weight.detach().cpu().double()
    kd = key[:, 0] - key[:, 1]
    return q, kd, model.learned_alpha_coefficient(), float(q @ kd / math.sqrt(model.d_head))


class BranchTrace(MatchedUpdateTrace):
    """Trace selected updates and compact exact allocation over every update."""

    def __init__(self, output, steps, start_step=0, norm_rows=None):
        super().__init__(output, steps)
        self.step = self.start_step = start_step
        self.norm_rows = copy.deepcopy(norm_rows or [])
        self.batch_digest = hashlib.sha256()
        self.batch_rows, self.allocation_rows = [], []
        self.window = {key: 0.0 for key in ("c_part", "score_part", "query_gap_part", "key_gap_part", "r_change")}
        self.total_allocation = self.window.copy()
        self.max_allocation_residual = 0.0
        self.extra_before = None

    def before_step(self, model, tokens, labels, loss, optimizer):
        self.mini_before = mini_state(model)
        self.batch_digest.update(str((self.step + 1, tuple(tokens.shape))).encode())
        self.batch_digest.update(tokens.detach().cpu().contiguous().numpy().tobytes())
        self.batch_digest.update(labels.detach().cpu().contiguous().numpy().tobytes())
        if self.step + 1 in self.steps:
            self.extra_before = {"weights": cpu_tree(model.state_dict()),
                                 "moments": named_optimizer_state(model, optimizer),
                                 "gradients": {name: cpu_tree(parameter.grad) for name, parameter in model.named_parameters()}}
        super().before_step(model, tokens, labels, loss, optimizer)

    def after_step(self, model, tokens, labels, loss, optimizer):
        q0, k0, c0, d0 = self.mini_before
        q1, k1, c1, d1 = mini_state(model)
        cpart, scorepart = (d0 + d1) / 2 * (c1 - c0), (c0 + c1) / 2 * (d1 - d0)
        parts = {"c_part": cpart, "score_part": scorepart,
                 "query_gap_part": float((q1 - q0) @ ((k0 + k1) / 2) / math.sqrt(model.d_head)),
                 "key_gap_part": float(((q0 + q1) / 2) @ (k1 - k0) / math.sqrt(model.d_head)),
                 "r_change": c1 * d1 - c0 * d0}
        self.max_allocation_residual = max(self.max_allocation_residual, abs(parts["r_change"] - cpart - scorepart))
        for key, value in parts.items():
            self.window[key] += value
            self.total_allocation[key] += value
        super().after_step(model, tokens, labels, loss, optimizer)
        if self.step in self.steps:
            row = self.rows[-1]
            before = self.extra_before
            moments = named_optimizer_state(model, optimizer)
            scalar = model.alpha_log_scale_unconstrained
            group = next(group for group in optimizer.param_groups if any(parameter is scalar for parameter in group["params"]))
            old = before["moments"].get("alpha_log_scale_unconstrained", {})
            new = moments["alpha_log_scale_unconstrained"]
            counter = float(new["step"])
            beta1, beta2 = group["betas"]
            direction = -float(new["exp_avg"]) / (1 - beta1 ** counter) / (
                math.sqrt(float(new["exp_avg_sq"]) / (1 - beta2 ** counter)) + group["eps"])
            row.update(coefficient_lr=group["lr"], u_step_before=float(old.get("step", 0)),
                       u_step_after=counter, u_moment_before=float(old.get("exp_avg", 0)),
                       u_moment_after=float(new["exp_avg"]), u_variance_before=float(old.get("exp_avg_sq", 0)),
                       u_variance_after=float(new["exp_avg_sq"]), u_normalized_direction=direction,
                       actual_u_update_over_lr=row["u_update"] / group["lr"],
                       optimizer_moments_before=json.dumps({name: {key: value.reshape(-1).tolist() for key, value in state.items()
                                                                  if isinstance(value, torch.Tensor)}
                                                           for name, state in before["moments"].items()}))
            for name in ("classifier.weight", "classifier.bias"):
                update = model.state_dict()[name].detach().cpu().double() - before["weights"][name].double()
                key = "head_weight" if name.endswith("weight") else "head_bias"
                row[f"{key}_actual_update"] = json.dumps(update.reshape(-1).tolist())
                row[f"{key}_actual_update_norm"] = float(update.norm())
            if self.step == self.start_step + 1:
                torch.save({"before": before, "after_weights": cpu_tree(model.state_dict()),
                            "after_moments": moments}, self.output / "first_update.pt")
            if self.step % 200 == 0 or self.step == max(self.steps):
                write_csv(self.output / "update_trace.csv", self.rows)
            self.extra_before = None
        if self.step % 200 == 0:
            self.batch_rows.append({"optimizer_updates": self.step, "tail_updates": self.step - self.start_step,
                                    "rolling_batch_sha256": self.batch_digest.hexdigest()})
            self.allocation_rows.append({"optimizer_updates": self.step, **self.window})
            self.window = {key: 0.0 for key in self.window}
            write_csv(self.output / "batch_progress.csv", self.batch_rows)
            write_csv(self.output / "allocation_by_200.csv", self.allocation_rows)


def first_epoch_batch(bundle, train):
    batches = [batch for loader in train for batch in loader]
    random.Random(bundle["config"]["seed"] + 20000 + bundle["next_epoch"]).shuffle(batches)
    return batches[0]


def forecast_first_update(bundle, device):
    model, optimizer, train, _ = restore_parent(bundle, device)
    split_coefficient_group(model, optimizer, 1.0)
    tokens, labels = first_epoch_batch(bundle, train)[:2]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.binary_cross_entropy_with_logits(model(tokens.to(device)), labels.to(device)).backward()
    predictions = []
    baseline_after = None
    for tag, scale in MULTIPLIERS:
        proposal, proposal_opt = copy.deepcopy((model, optimizer))
        for (_, original), (_, cloned) in zip(model.named_parameters(), proposal.named_parameters()):
            cloned.grad = original.grad.clone()
        proposal_opt.param_groups[1]["lr"] = bundle["config"]["learning_rate"] * scale
        proposal_opt.step()
        state = trace_state(proposal)
        predictions.append({"tag": tag, "multiplier": scale, "u_after": state["u"], "c_after": state["c"],
                            "delta_after": state["delta"], "r_after": state["c"] * state["delta"]})
        if tag == "control":
            baseline_after = cpu_tree(proposal.state_dict())
    assert all(item["delta_after"] == predictions[0]["delta_after"] for item in predictions)
    for item in predictions:
        item["r_minus_control"] = item["r_after"] - next(row["r_after"] for row in predictions if row["tag"] == "control")
    return {"kind": "Discarded one-step AdamW calculations from identical saved states and first batch",
            "parent_sha256": tree_digest(bundle), "predictions": predictions,
            "first_batch_sha256": tree_digest((tokens, labels)), "control_after_state": baseline_after}


def short_parent_checks(bundle, config, device, output, steps=32):
    output.mkdir()
    target = replace(config, max_train_steps=bundle["optimizer_updates"] + steps)
    results = []
    for name, multiplier in (("original_group", None), ("split_group", 1.0)):
        directory = output / name
        directory.mkdir()
        recorder = restored_recorder(bundle, target, directory)
        trace = BranchTrace(directory, {bundle["optimizer_updates"] + 1}, bundle["optimizer_updates"], bundle["norm_rows"])
        result = continue_training(bundle, target, device, directory, multiplier, recorder, trace)
        results.append((result, trace, capture_rng()))
    a, b = results[0][0], results[1][0]
    assert state_digest(a[0]) == state_digest(b[0])
    assert tree_digest(named_optimizer_state(a[0], a[1])) == tree_digest(named_optimizer_state(b[0], b[1]))
    assert tree_digest(loader_state(a[2])) == tree_digest(loader_state(b[2]))
    assert (output / "original_group/train_history.csv").read_bytes() == (output / "split_group/train_history.csv").read_bytes()
    assert results[0][1].batch_digest.hexdigest() == results[1][1].batch_digest.hexdigest()
    assert tree_digest(results[0][2]) == tree_digest(results[1][2])
    return {"steps": steps, "weights_equal": True, "moments_equal": True, "loaders_equal": True,
            "history_equal": True, "batch_hash_equal": True, "rng_equal": True}


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    parser.add_argument("--wall-time-minutes", type=float, default=60)
    args = parser.parse_args()
    output = project_dir() / args.output_dir
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    for name in ("parents", "validation", "branches", "source_snapshot"):
        (output / name).mkdir()
    device = resolve_device(args.device)
    baseline = project_dir() / "runs/training_length/seed_replication_20260929_01"
    repo = project_dir().parent
    manifest = {"date": "2026-10-10", "launch_arguments": vars(args),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True),
        "python": platform.python_version(), "torch": torch.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda": torch.version.cuda, "seeds": [3, 4, 6], "lengths": [10, 100], "multipliers": dict(MULTIPLIERS)}
    write_json(output / "manifest.json", manifest)
    for relative in ("src/run_coefficient_lr_branches.py", "src/stage3_simplified_attention.py",
                     "src/run_training_length_pilot.py", "src/run_training_length_trace.py",
                     "src/run_matched_initial_control.py", "src/run_training_length_init.py",
                     "src/analyze_stage3_mechanism.py", "documents/training_length/COEFFICIENT_LR_BRANCH_PLAN.md",
                     "documents/training_length/COEFFICIENT_LR_BRANCH_SPEC.md"):
        source = project_dir() / relative
        shutil.copy2(source, output / "source_snapshot" / source.name)
    (output / "source_snapshot/working_diff.patch").write_text(
        subprocess.check_output(["git", "diff"], cwd=repo, text=True), encoding="utf-8")
    started = time.perf_counter()
    parents, rows = [], []
    def remaining():
        return args.wall_time_minutes * 60 - (time.perf_counter() - started)
    for seed in (3, 4, 6):
        for length in (10, 100):
            assert remaining() > 120, "Time cap before parent capture"
            name = f"n{length}_s{seed}"
            directory = output / "parents" / name
            directory.mkdir()
            reference = baseline / f"learned_log_n{length}_s{seed}"
            original = json.loads((reference / "config.json").read_text())
            settings = {key: value for key, value in original.items() if key in Stage3Config.__dataclass_fields__}
            settings["train_lengths"] = tuple(settings["train_lengths"])
            settings["eval_lengths"] = tuple(settings["eval_lengths"])
            config = replace(Stage3Config(**settings), device=args.device,
                             output_dir=directory.relative_to(project_dir()).as_posix())
            prefix = replace(config, max_train_steps=1600)
            recorder = TrainingRecorder(prefix, directory, 200)
            trace = BranchTrace(directory, PREFIX_STEPS)
            def initialize(model):
                recorder(model, None, None)
                trace.record_norms(model, 0)
            def observe(model, tokens, labels):
                if tokens is None:
                    initialize(model)
                else:
                    recorder(model, tokens, labels)
            captured = []
            def boundary(model, optimizer, train, val, epoch, updates, history):
                if updates == 1600:
                    bundle = capture_parent(config, model, optimizer, train, val, epoch, updates, history, recorder, trace)
                    torch.save(bundle, directory / "parent_state.pt")
                    captured.append(bundle)
            print(f"PARENT {name}", flush=True)
            set_reproducibility(seed)
            model, updates = train_model(prefix, device=device, output_dir=directory,
                training_observer=observe, update_observer=trace, epoch_observer=boundary)
            assert updates == 1600 and len(captured) == 1
            bundle = captured[0]
            assert recorder.initial_digest == original["initial_state_sha256"]
            assert read_rows(directory / "training_diagnostics.csv") == [row for row in read_rows(reference / "training_diagnostics.csv")
                                                                        if int(row["optimizer_updates"]) <= 1600]
            assert read_rows(directory / "train_history.csv") == [row for row in read_rows(reference / "train_history.csv")
                                                                 if int(row["optimizer_updates"]) <= 1600]
            metrics = initial_metrics(model)
            tokens, labels = fixed_inputs(length, device)
            with torch.no_grad():
                logits = model(tokens)
                assert (logits[:3] > 0).all() and logits[-1] < 0
            assert metrics["head_slope"] > 0 and metrics["delta"] > 0
            write_json(directory / "parent_verification.json", {"prefix_diagnostics_exact": True,
                "prefix_history_exact": True, "initial_hash_equal": True, "own_classes_correct": True,
                "parent_state_sha256": tree_digest(bundle), "checkpoint_file_sha256": hashlib.sha256((directory / "parent_state.pt").read_bytes()).hexdigest(),
                "positive_head_and_gap": True, "branch_metrics": metrics})
            check = short_parent_checks(bundle, config, device, output / "validation" / name)
            write_json(output / "validation" / name / "group_verification.json", check)
            forecast = forecast_first_update(bundle, device)
            torch.save(forecast, directory / "first_step_forecast.pt")
            write_json(directory / "first_step_forecast.json", {key: value for key, value in forecast.items() if key != "control_after_state"})
            parents.append((config, name, bundle, reference, forecast))
            print(f"READY {name} r={metrics['c_delta']:.6f}", flush=True)
    write_json(output / "predictions_frozen.json", {"all_parent_forecasts_saved": True,
        "before_comparison_tails": True, "parents": [name for _, name, _, _, _ in parents],
        "elapsed_seconds": time.perf_counter() - started,
        "hypotheses": ["r_fast > r_control > r_slow", "absolute effect larger at length 10",
                       "G_fast > G_control > G_slow at 6400 and common BCE 0.01", "fitting retained"]})
    for config, name, bundle, reference, forecast in parents:
        sibling_first, sibling_batches = [], []
        for tag, multiplier in MULTIPLIERS:
            assert remaining() > max((row["wall_seconds"] * 1.2 for row in rows), default=120), "Time cap before branch"
            directory = output / "branches" / f"{name}_{tag}"
            directory.mkdir()
            config = replace(config, output_dir=directory.relative_to(project_dir()).as_posix())
            recorder = restored_recorder(bundle, config, directory)
            trace = BranchTrace(directory, TAIL_STEPS, 1600, bundle["norm_rows"])
            branch_started = time.perf_counter()
            print(f"BRANCH {len(rows) + 1}/18 {directory.name}", flush=True)
            model, optimizer, train, val = continue_training(bundle, config, device, directory, multiplier, recorder, trace)
            assert len(trace.rows) == 232 and trace.step == 6400
            assert max(trace.max_residual.values()) < 1e-10 and trace.max_allocation_residual < 1e-10
            torch.save({"named_optimizer": named_optimizer_state(model, optimizer),
                        "optimizer": cpu_tree(optimizer.state_dict()), "rng": capture_rng(),
                        "train_loaders": loader_state(train), "val_loaders": loader_state(val),
                        "capture_boundary": "after final epoch validation, before final evaluation"}, directory / "final_continuation_state.pt")
            row = finish_run(config, directory, device, model, 6400, recorder, branch_started)
            first = torch.load(directory / "first_update.pt", map_location="cpu", weights_only=False)
            sibling_first.append(first)
            sibling_batches.append(trace.batch_rows)
            prediction = next(item for item in forecast["predictions"] if item["tag"] == tag)
            assert trace.rows[0]["c_delta_after"] == prediction["r_after"]
            if tag == "control":
                assert tree_digest(first["after_weights"]) == tree_digest(forecast["control_after_state"])
                write_json(directory / "baseline_verification.json", verify_replay(directory, reference))
            verification = {"parent_state_sha256": tree_digest(bundle), "lr_multiplier": multiplier,
                "first_update_prediction_exact": True, "tail_batch_sha256": trace.batch_digest.hexdigest(),
                "tail_updates": 4800, "traced_updates": 232,
                "max_decomposition_residual": max(max(trace.max_residual.values()), trace.max_allocation_residual),
                "total_allocation": trace.total_allocation}
            write_json(directory / "branch_verification.json", verification)
            row.update(parent=name, multiplier=multiplier, tag=tag, traced_updates=232,
                       tail_batch_sha256=trace.batch_digest.hexdigest())
            rows.append(row)
            write_csv(output / "summary.csv", rows)
            write_json(output / "status.json", {"completed_parents": len(parents), "completed_branches": len(rows),
                "planned_branches": 18, "wall_seconds": time.perf_counter() - started,
                "complete": False, "scheduled_updates": 96000,
                "group_check_updates": 6 * 2 * 32, "forecast_proposal_updates": 6 * 3})
            print(f"DONE {directory.name} seconds={row['wall_seconds']:.2f} r={row['c_delta']:.6f}", flush=True)
        assert all(tree_digest(item) == tree_digest(sibling_batches[0]) for item in sibling_batches)
        first = sibling_first[0]
        for item in sibling_first[1:]:
            assert tree_digest(item["before"]) == tree_digest(first["before"])
            assert tree_digest(item["after_moments"]) == tree_digest(first["after_moments"])
            for key in first["after_weights"]:
                if key != "alpha_log_scale_unconstrained":
                    assert torch.equal(item["after_weights"][key], first["after_weights"][key])
        write_json(output / "parents" / name / "sibling_verification.json", {
            "initial_states_and_gradients_exact": True, "first_query_key_head_updates_exact": True,
            "first_moments_exact": True, "all_tail_batch_hashes_exact": True})
        parent_directory = output / "parents" / name
        saved = json.loads((parent_directory / "parent_verification.json").read_text())
        assert hashlib.sha256((parent_directory / "parent_state.pt").read_bytes()).hexdigest() == saved["checkpoint_file_sha256"]
    write_json(output / "status.json", {"completed_parents": 6, "completed_branches": 18,
        "planned_branches": 18, "wall_seconds": time.perf_counter() - started,
        "complete": True, "all_sibling_checks_passed": True, "scheduled_updates": 96000,
        "group_check_updates": 384, "forecast_proposal_updates": 18})
    print("COMPLETE 18 branches", flush=True)


if __name__ == "__main__":
    main()
