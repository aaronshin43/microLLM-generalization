"""Continue nine baseline states with original optimizer groups and full data progress."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from analyze_training_length_sweep import analytic, is_fitted
from run_coefficient_lr_branches import (capture_parent, capture_rng, cpu_tree, loader_state,
    make_loaders, named_optimizer_state, read_rows, restore_loaders, restore_rng, tree_digest)
from run_coefficient_lr_holdout import file_digest
from run_training_length_pilot import TrainingRecorder, finish_run, mechanism
from run_training_length_sweep import COEFFICIENT, EVAL_LENGTHS, CoefficientTrace, config_from_record
from run_training_length_trace import trace_state
from stage3_simplified_attention import (SimplifiedLastQueryAttentionClassifier, project_dir,
    resolve_device, run_loaders_once, set_reproducibility, write_csv, write_json)

SEEDS = (3, 4, 6)
LENGTHS = (10, 30, 1000)
START, END = 6400, 12800


def parameter_groups(bundle):
    """Map saved integer optimizer IDs to names, checking every saved moment tensor."""
    names = bundle["parameter_names"]
    groups = bundle["optimizer"]["param_groups"]
    if "optimizer_group_parameter_names" in bundle:
        mapping = bundle["optimizer_group_parameter_names"]
    elif len(groups) == 1:
        mapping = [names]
    else:
        assert len(groups) == 2 and len(groups[1]["params"]) == 1
        mapping = [[name for name in names if name != COEFFICIENT], [COEFFICIENT]]
    assert len(groups) == len(mapping)
    assert sorted(name for group in mapping for name in group) == sorted(names)
    for group, group_names in zip(groups, mapping):
        assert len(group["params"]) == len(group_names)
        for identifier, name in zip(group["params"], group_names):
            assert tree_digest(bundle["optimizer"]["state"][identifier]) == tree_digest(bundle["named_optimizer"][name]), name
    return copy.deepcopy(mapping)


def restore_start(bundle, device):
    config = config_from_record(bundle["config"])
    train, val = make_loaders(config)
    model = SimplifiedLastQueryAttentionClassifier(d_head=config.d_head, alpha_mode=config.alpha_mode,
        alpha_log_scale_init=config.alpha_log_scale_init, target_token_count=config.target_token_count,
        non_target_token_count=config.non_target_token_count).to(device)
    model.load_state_dict(bundle["model"])
    model.train(bundle["model_training"])
    named = dict(model.named_parameters())
    assert list(named) == bundle["parameter_names"]
    mapping = parameter_groups(bundle)
    groups = [{**{key: copy.deepcopy(value) for key, value in group.items() if key != "params"},
        "params": [named[name] for name in group_names]}
        for group, group_names in zip(bundle["optimizer"]["param_groups"], mapping)]
    optimizer = torch.optim.AdamW(groups, lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer.load_state_dict(copy.deepcopy(bundle["optimizer"]))
    assert tree_digest(cpu_tree(optimizer.state_dict())) == tree_digest(bundle["optimizer"])
    assert tree_digest(named_optimizer_state(model, optimizer)) == tree_digest(bundle["named_optimizer"])
    assert tree_digest(model.state_dict()) == tree_digest(bundle["model"])
    restore_loaders(train, bundle["train_loaders"])
    restore_loaders(val, bundle["val_loaders"])
    restore_rng(bundle["rng"])
    assert tree_digest(capture_rng()) == tree_digest(bundle["rng"])
    return model, optimizer, train, val


def expand_prefix(rows):
    """Add only absent diagnostic lengths from saved parameters; retain original rows."""
    expanded = []
    for step in sorted({int(row["optimizer_updates"]) for row in rows}):
        group = {int(row["length"]): row for row in rows if int(row["optimizer_updates"]) == step}
        base = group[100]
        values = {key: float(base[key]) if base[key] not in (None, "") else None
            for key in ("c", "delta", "c_delta", "head_slope", "head_intercept", "head_threshold")}
        for length in EVAL_LENGTHS:
            expanded.append(copy.deepcopy(group[length]) if length in group else
                {**base, "length": length, **analytic(values, length)})
    return expanded


def assemble_start(root, length, seed):
    name = f"n{length}_s{seed}"
    source_files = []
    if length == 10:
        directory = root / "runs/training_length/coefficient_lr_branches_20261010_01/branches" / f"{name}_control"
        parent_path = directory.parent.parent / "parents" / name / "parent_state.pt"
        source_files += [directory / filename for filename in ("final_continuation_state.pt", "model.pt", "config.json",
            "train_history.csv", "training_diagnostics.csv", "exposure.json", "baseline_verification.json")]
        source_files.append(parent_path)
        state = torch.load(directory / "final_continuation_state.pt", map_location="cpu", weights_only=False)
        weights = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
        config_record = json.loads((directory / "config.json").read_text())
        parent = torch.load(parent_path, map_location="cpu", weights_only=False)
        historical = root / f"runs/training_length/seed_replication_20260929_01/learned_log_{name}"
        source_files += [historical / filename for filename in ("model.pt", "config.json", "train_history.csv", "training_diagnostics.csv")]
        original = torch.load(historical / "model.pt", map_location="cpu", weights_only=False)
        assert weights["optimizer_updates"] == original["optimizer_updates"] == START
        assert tree_digest(weights["state_dict"]) == tree_digest(original["state_dict"])
        assert all(json.loads((directory / "baseline_verification.json").read_text()).values())
        for filename in ("train_history.csv", "training_diagnostics.csv"):
            assert read_rows(directory / filename) == read_rows(historical / filename)
        history = [{key: int(value) if key in ("epoch", "optimizer_updates", "updates_this_epoch") else float(value)
            for key, value in row.items()} for row in read_rows(directory / "train_history.csv")]
        assert history[-1]["epoch"] == 200 and history[-1]["optimizer_updates"] == START
        assert state["capture_boundary"] == "after final epoch validation, before final evaluation"
        bundle = {"config": asdict(config_from_record(config_record)), "model": weights["state_dict"],
            "optimizer": state["optimizer"], "named_optimizer": state["named_optimizer"],
            "parameter_names": parent["parameter_names"], "model_training": False,
            "epoch": 200, "optimizer_updates": START, "next_epoch": 201, "batch_cursor": 0,
            "validation_complete": True, "history": history,
            "train_loaders": state["train_loaders"], "val_loaders": state["val_loaders"], "rng": state["rng"],
            "recorder": {"updates": START, "exposure": json.loads((directory / "exposure.json").read_text())["by_length"],
                "initial_digest": config_record["initial_state_sha256"],
                "rows": expand_prefix(read_rows(directory / "training_diagnostics.csv"))},
            "norm_rows": [], "scheduler": None, "scaler": None}
        provenance = {"format": "assembled_model_and_two_group_continuation_state",
            "mode_source": "False from documented completed-validation capture boundary",
            "missing_diagnostic_lengths_derived_from_parameters": [30, 300],
            "earlier_complete_gradient_history": False}
    else:
        directory = root / "runs/training_length/length_sweep_20261024_01/new_runs" / name
        source_files = [directory / filename for filename in ("state_step6400.pt", "config.json", "model.pt",
            "training_diagnostics.csv", "train_history.csv", "coefficient_trace.csv", "verification.json", "exposure.json")]
        bundle = torch.load(directory / "state_step6400.pt", map_location="cpu", weights_only=False)
        verification = json.loads((directory / "verification.json").read_text())
        assert file_digest(directory / "state_step6400.pt") == verification["checkpoint_sha256"][str(START)]
        provenance = {"format": "single_full_one_group_bundle", "earlier_complete_gradient_history": True,
            "earlier_coefficient_trace": str(directory / "coefficient_trace.csv"), "missing_diagnostic_lengths_derived_from_parameters": []}
    assert bundle["optimizer_updates"] == bundle["recorder"]["updates"] == START
    assert bundle["epoch"] == 200 and bundle["next_epoch"] == 201 and bundle["batch_cursor"] == 0
    assert bundle["validation_complete"] and not bundle["model_training"]
    assert len(bundle["history"]) == 200 and len(bundle["recorder"]["rows"]) == 33 * 6
    assert all(int(state["step"]) == START for state in bundle["named_optimizer"].values())
    assert sum(item["sequences"] for item in bundle["recorder"]["exposure"].values()) == 409600
    assert all(group["lr"] == .003 and group["weight_decay"] == 0 and tuple(group["betas"]) == (.9, .999)
        and group["eps"] == 1e-8 for group in bundle["optimizer"]["param_groups"])
    bundle["optimizer_group_parameter_names"] = parameter_groups(bundle)
    return bundle, {"length": length, "seed": seed, "source_directory": str(directory), **provenance,
        "source_sha256": {str(path): file_digest(path) for path in source_files},
        "optimizer_group_parameter_names": bundle["optimizer_group_parameter_names"],
        "optimizer_options": [{key: value for key, value in group.items() if key != "params"} for group in bundle["optimizer"]["param_groups"]]}


class BatchWitness:
    """Capture batch identities for discarded serialization/passivity comparisons."""
    def __init__(self, start, observer):
        self.step, self.observer, self.batches = start, observer, []

    def before_step(self, model, tokens, labels, loss, optimizer):
        self.step += 1
        self.batches.append(tree_digest((self.step, tokens, labels)))
        if self.observer is not None:
            self.observer.before_step(model, tokens, labels, loss, optimizer)

    def after_step(self, *args):
        if self.observer is not None:
            self.observer.after_step(*args)


def restored_recorder(bundle, config, output):
    recorder = TrainingRecorder(config, output, 200, diagnostic_lengths=EVAL_LENGTHS)
    for key, value in bundle["recorder"].items():
        setattr(recorder, key, copy.deepcopy(value))
    write_csv(output / "training_diagnostics.csv", recorder.rows)
    return recorder


def continue_path(bundle, config, device, output, *, observed=True, witness=False, exports=(), deadline=None):
    model, optimizer, train, val = restore_start(bundle, device)
    recorder = restored_recorder(bundle, config, output)
    trace = CoefficientTrace(output, config.max_train_steps, deadline, start_step=bundle["optimizer_updates"]) if observed else None
    observer = BatchWitness(bundle["optimizer_updates"], trace) if witness else trace
    history = copy.deepcopy(bundle["history"])
    updates, epoch = bundle["optimizer_updates"], bundle["epoch"]
    criterion = torch.nn.BCEWithLogitsLoss()
    exported = {}
    while updates < config.max_train_steps:
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("Continuation wall-time ceiling reached")
        epoch += 1
        train_loss, metrics, added = run_loaders_once(model, train, criterion=criterion, device=device,
            optimizer=optimizer, max_steps=config.max_train_steps - updates,
            batch_shuffle_seed=config.seed + 20000 + epoch,
            training_observer=recorder if observed else None, update_observer=observer)
        updates += added
        val_loss, val_metrics, _ = run_loaders_once(model, val, criterion=criterion, device=device)
        history.append({"epoch": epoch, "optimizer_updates": updates, "updates_this_epoch": added,
            "train_loss": train_loss, "train_accuracy": metrics["accuracy"],
            "train_positive_accuracy": metrics["positive_accuracy"], "train_negative_accuracy": metrics["negative_accuracy"],
            "val_loss": val_loss, "val_accuracy": val_metrics["accuracy"],
            "val_positive_accuracy": val_metrics["positive_accuracy"], "val_negative_accuracy": val_metrics["negative_accuracy"]})
        if updates in exports:
            assert observed
            state = capture_parent(config, model, optimizer, train, val, epoch, updates, history, recorder, trace)
            state["optimizer_group_parameter_names"] = parameter_groups(state)
            torch.save(state, output / f"state_step{updates}.pt")
            exported[str(updates)] = file_digest(output / f"state_step{updates}.pt")
        write_csv(output / "train_history.csv", history)
    snapshot = {"model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
        "named_optimizer": named_optimizer_state(model, optimizer), "rng": capture_rng(),
        "train_loaders": loader_state(train), "val_loaders": loader_state(val),
        "history": history, "updates": updates, "epoch": epoch}
    return model, recorder, trace, snapshot, exported, observer.batches if witness else None


def serialization_probe(bundle, device, output):
    output.mkdir()
    config = config_from_record(bundle["config"])
    results, batches = [], []
    for tag, steps, observed, source in (("uninterrupted_unobserved", 6464, False, bundle),
        ("first32_observed", 6432, True, bundle)):
        directory = output / tag
        directory.mkdir()
        target = replace(config, max_train_steps=steps, output_dir=str(directory))
        result = continue_path(source, target, device, directory, observed=observed, witness=True,
            exports=(6432,) if observed else ())
        results.append(result[3])
        batches.append(result[5])
    directory = output / "second32_observed"
    directory.mkdir()
    saved = torch.load(output / "first32_observed/state_step6432.pt", map_location="cpu", weights_only=False)
    result = continue_path(saved, replace(config, max_train_steps=6464), device, directory, witness=True)
    assert tree_digest(results[0]) == tree_digest(result[3]), "Serialized/observed continuation diverged"
    assert batches[0] == batches[1] + result[5], "Probe batch sequence diverged"
    return {"length": config.train_length, "seed": config.seed, "start_step": START, "end_step": 6464,
        "discarded_updates": 128, "weights_optimizer_rng_loaders_history_exact": True, "batch_sequence_exact": True,
        "observed_split_vs_unobserved_uninterrupted_exact": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--wall-time-minutes", default=60, type=float)
    args = parser.parse_args()
    root, device = project_dir(), resolve_device(args.device)
    output = root / args.output_dir
    if output.exists():
        raise FileExistsError(output)
    for folder in ("starts", "validation", "runs", "source_snapshot"):
        (output / folder).mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + 60 * args.wall_time_minutes
    set_reproducibility(0)
    inventory, starts, initial_hashes = [], {}, {}
    for length in LENGTHS:
        for seed in SEEDS:
            name = f"n{length}_s{seed}"
            bundle, provenance = assemble_start(root, length, seed)
            model, optimizer, train, val = restore_start(bundle, device)
            values = mechanism(model)
            direct_tokens = torch.ones((2, length), dtype=torch.long, device=device)
            direct_tokens[0, 0] = 0
            with torch.no_grad():
                logits = model(direct_tokens).detach().cpu().tolist()
            expected = analytic(values, length)
            assert logits[0] > 0 > logits[1]
            assert abs(logits[0] - expected["positive_logit"]) < 3e-5
            assert abs(logits[1] - expected["negative_logit"]) < 3e-5
            reference = next(row for row in bundle["recorder"]["rows"] if int(row["optimizer_updates"]) == START and int(row["length"]) == 100)
            for key in ("c", "delta", "c_delta", "head_slope", "head_intercept", "head_threshold"):
                assert abs(values[key] - float(reference[key])) < 1e-10
            assert abs(analytic(values, 100)["balanced_bce"] - float(reference["balanced_bce"])) < 1e-10
            initial_hashes.setdefault(seed, bundle["recorder"]["initial_digest"])
            assert initial_hashes[seed] == bundle["recorder"]["initial_digest"]
            assert tree_digest(capture_rng()) == tree_digest(bundle["rng"])
            path = output / "starts" / f"{name}.pt"
            torch.save(bundle, path)
            provenance.update(name=name, normalized_start_sha256=file_digest(path),
                scientific_state_sha256=tree_digest({k: bundle[k] for k in ("model", "optimizer", "named_optimizer", "rng", "train_loaders", "val_loaders")}),
                restored_weights_moments_groups_rng_loaders_exact=True, successful_own_fit=True,
                starting_parameters=values, own_bce=expected["balanced_bce"], common100_bce=float(reference["balanced_bce"]))
            inventory.append(provenance)
            starts[name] = bundle
            print(f"AUDIT {name} groups={len(optimizer.param_groups)} r={values['c_delta']:.6f}", flush=True)
    write_json(output / "source_inventory.json", {"cells": inventory})
    checks = []
    for length in LENGTHS:
        check = serialization_probe(starts[f"n{length}_s3"], device, output / "validation" / f"n{length}_s3")
        checks.append(check)
        write_json(output / "serialization_checks.json", {"checks": checks})
        print(f"CHECK length={length} exact=True", flush=True)
    sources = (Path(__file__).name, "run_training_length_sweep.py", "analyze_training_length_sweep.py",
        "run_training_length_pilot.py", "run_training_length_trace.py", "run_coefficient_lr_branches.py",
        "run_coefficient_lr_holdout.py", "stage3_simplified_attention.py", "analyze_stage3_mechanism.py")
    for name in sources:
        shutil.copy2(root / "src" / name, output / "source_snapshot" / name)
    for name in ("LENGTH_EXTENSION_12800_PLAN.md", "LENGTH_EXTENSION_12800_SPEC.md"):
        shutil.copy2(root / "documents/training_length" / name, output / "source_snapshot" / name)
    manifest = {"date": "2026-10-24", "arguments": vars(args), "lengths": LENGTHS, "seeds": SEEDS,
        "run_order": list(starts), "start_step": START, "end_step": END, "common100_bce_targets": [.001, .0005, .0001],
        "primary_loss_target": .001, "source_inventory_sha256": file_digest(output / "source_inventory.json"),
        "source_sha256": {name: file_digest(output / "source_snapshot" / name) for name in sources},
        "plan_sha256": file_digest(output / "source_snapshot/LENGTH_EXTENSION_12800_PLAN.md"),
        "spec_sha256": file_digest(output / "source_snapshot/LENGTH_EXTENSION_12800_SPEC.md"),
        "serialization_checks": checks, "planned_new_scientific_updates": 57600, "discarded_updates": 384,
        "expected_new_sequence_exposures": 3686400, "expected_new_token_exposures": 1277952000,
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "dtype": str(torch.get_default_dtype()),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root.parent, text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--short"], cwd=root.parent, text=True),
        "frozen_before_scientific_tails": True, "hypotheses": ["retained fitting", "r10>r30>r1000 at12800", "longer-length crossing open"]}
    write_json(output / "manifest.json", manifest)
    rows = []
    status = {"complete": False, "planned_runs": 9, "completed_runs": 0, "new_scientific_updates": 0, "discarded_updates": 384}
    try:
        for name, bundle in starts.items():
            if time.perf_counter() >= deadline:
                raise TimeoutError("Wall-time ceiling before next continuation")
            directory = output / "runs" / name
            directory.mkdir()
            source_config = config_from_record(bundle["config"])
            config = replace(source_config, max_train_steps=END, epochs=400, eval_lengths=EVAL_LENGTHS,
                output_dir=directory.relative_to(root).as_posix())
            status.update(active_run=name, phase="scientific_continuation")
            write_json(output / "status.json", status)
            write_json(directory / "config.json", asdict(config))
            run_started = time.perf_counter()
            print(f"RUN {len(rows)+1}/9 {name}", flush=True)
            model, recorder, trace, snapshot, exported, _ = continue_path(bundle, config, device, directory,
                exports=(9600, END), deadline=deadline)
            assert len(trace.rows) == 6400 and len(trace.blocks) == 32 and trace.step == END
            assert [int(r["optimizer_updates"]) for r in read_rows(directory / "coefficient_trace.csv")] == list(range(6401, END + 1))
            assert snapshot["epoch"] == 400 and len(snapshot["history"]) == 400
            assert tree_digest(snapshot["history"][:200]) == tree_digest(bundle["history"])
            row = finish_run(config, directory, device, model, END, recorder, run_started, explicit_lengths=EVAL_LENGTHS)
            assert row["total_sequences"] == 819200 and row["total_tokens"] == 819200 * config.train_length
            new_exposure = {key: recorder.exposure[str(config.train_length)][key] - bundle["recorder"]["exposure"][str(config.train_length)][key]
                for key in ("sequences", "tokens", "positives", "negatives")}
            assert new_exposure["sequences"] == 409600 and new_exposure["positives"] == new_exposure["negatives"] == 204800
            assert all(int(state["step"]) == END for state in snapshot["named_optimizer"].values())
            assert len(snapshot["optimizer"]["param_groups"]) == len(bundle["optimizer"]["param_groups"])
            write_json(directory / "verification.json", {"start_state_restored_exact": True,
                "source_history_preserved_exact": True, "first_epoch": 201, "last_epoch": 400,
                "start_step": START, "end_step": END, "new_updates": 6400, "trace_rows": len(trace.rows),
                "max_accounting_residual": max(abs(r["r_residual"]) for r in trace.rows),
                "checkpoint_sha256": exported, "new_exposure": new_exposure,
                "optimizer_group_parameter_names": parameter_groups({**bundle, "optimizer": snapshot["optimizer"], "named_optimizer": snapshot["named_optimizer"]})})
            row.update(new_sequence_exposures=new_exposure["sequences"], new_token_exposures=new_exposure["tokens"])
            rows.append(row)
            write_csv(output / "summary.csv", rows)
            status.update(completed_runs=len(rows), new_scientific_updates=6400 * len(rows),
                new_sequence_exposures=sum(r["new_sequence_exposures"] for r in rows),
                new_token_exposures=sum(r["new_token_exposures"] for r in rows), wall_seconds=time.perf_counter() - started)
            write_json(output / "status.json", status)
            print(f"DONE {name} r={row['c_delta']:.6f} seconds={row['wall_seconds']:.1f}", flush=True)
        for item in inventory:
            for path, digest in item["source_sha256"].items():
                assert file_digest(Path(path)) == digest, "Historical source changed"
            assert file_digest(output / "starts" / f"{item['name']}.pt") == item["normalized_start_sha256"]
        assert status["new_sequence_exposures"] == 3686400 and status["new_token_exposures"] == 1277952000
        status.update(complete=True, phase="complete", active_run=None, historical_sources_unchanged=True,
            wall_seconds=time.perf_counter() - started)
        write_json(output / "status.json", status)
        print("COMPLETE nine 6400-to-12800 baseline continuations", flush=True)
    except Exception as error:
        status.update(phase="incomplete", error=f"{type(error).__name__}: {error}", wall_seconds=time.perf_counter() - started)
        write_json(output / "status.json", status)
        raise


if __name__ == "__main__":
    main()
