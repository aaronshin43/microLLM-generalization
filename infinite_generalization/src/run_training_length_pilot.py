"""Run the bounded binary-detection pilot using the existing Stage 3 training path."""

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
from dataclasses import asdict
from pathlib import Path

import torch

from analyze_stage3_mechanism import analyze_model
from stage3_simplified_attention import (
    Stage3Config, add_train_delta_theory, evaluate_length, project_dir, resolve_device,
    save_model_checkpoint, set_reproducibility, train_model, write_csv, write_json,
)

LENGTH_SETS = [(10,), (100,), (10, 100)]
MODES = ["constant", "log", "learned_log"]
EXPLICIT_LENGTHS = (10, 100, 1000, 10000)
ANALYTICAL_LENGTHS = (*EXPLICIT_LENGTHS, 100000, 1000000, 10000000, 10**10, 10**20, 10**30)


def state_digest(model):
    """Hash all initial state elements for paired-initialization verification."""
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sigmoid(value):
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


def mechanism(model):
    analysis = analyze_model(model)
    wt, wu = model.classifier.weight.detach().cpu()[0].tolist()
    bias = model.classifier.bias.item()
    slope = wt - wu
    coefficient = model.learned_alpha_coefficient()
    return {
        "delta": analysis["delta"], "target_score": analysis["target_score_a"],
        "non_target_score": analysis["non_target_score_b"],
        "c": coefficient if math.isfinite(coefficient) else None,
        "c_delta": coefficient * analysis["delta"] if math.isfinite(coefficient) else None,
        "head_slope": slope, "head_intercept": wu + bias,
        "head_threshold": -(wu + bias) / slope if slope != 0 else None,
    }


def attention_gap(mode, values, log_length):
    """Evaluate the exact two-score log-odds without constructing a sequence."""
    log_competitors = log_length + math.log1p(-math.exp(-log_length))
    if mode == "constant":
        alpha = 1.0
    elif mode == "log":
        alpha = log_length
    else:
        alpha = 1 + values["c"] * (log_length + math.log1p(math.exp(-log_length)))
    return alpha * values["delta"] - log_competitors, alpha


def predicted_failure_log10(mode, values):
    """Find a descending positive-class threshold crossing under the exact model."""
    exponent = 0 if mode == "constant" else (values["delta"] if mode == "log" else values["c_delta"])
    threshold = values["head_threshold"]
    if values["head_slope"] <= 0 or threshold is None or not 0 < threshold < 1 or exponent >= 1:
        return None
    target_gap = math.log(threshold) - math.log1p(-threshold)
    low = math.log(2)
    if attention_gap(mode, values, low)[0] <= target_gap:
        return math.log10(2)
    high = max(2.0, low * 2)
    while attention_gap(mode, values, high)[0] > target_gap:
        high *= 2
    for _ in range(100):
        middle = (low + high) / 2
        if attention_gap(mode, values, middle)[0] > target_gap:
            low = middle
        else:
            high = middle
    return (low + high) / (2 * math.log(10))


class TrainingRecorder:
    """Record passive diagnostics and realized exposure after optimizer updates."""

    def __init__(self, config, output, interval, reference_run=None, verify_step=1600):
        self.config, self.output, self.interval = config, output, interval
        self.updates = 0
        self.exposure = {str(length): {"sequences": 0, "tokens": 0, "positives": 0, "negatives": 0}
                         for length in config.train_lengths}
        self.initial_digest = None
        self.rows = []
        self.reference_run = reference_run
        self.verify_step = verify_step

    def __call__(self, model, tokens, labels):
        if tokens is None:
            self.initial_digest = state_digest(model)
            if self.reference_run is not None:
                reference_config = json.loads((self.reference_run / "config.json").read_text())
                assert self.initial_digest == reference_config["initial_state_sha256"], "Pilot initialization mismatch"
        else:
            self.updates += 1
            count, length = tokens.shape
            positives = int(labels.sum().item())
            entry = self.exposure[str(length)]
            entry["sequences"] += count
            entry["tokens"] += count * length
            entry["positives"] += positives
            entry["negatives"] += count - positives
        if tokens is None or self.updates % self.interval == 0 or self.updates == self.config.max_train_steps:
            values = mechanism(model)
            totals = {key: sum(entry[key] for entry in self.exposure.values())
                      for key in ["sequences", "tokens", "positives", "negatives"]}
            for length in EXPLICIT_LENGTHS:
                gap, alpha = attention_gap(self.config.alpha_mode, values, math.log(length))
                mass = sigmoid(gap)
                positive_logit = values["head_slope"] * mass + values["head_intercept"]
                negative_logit = values["head_intercept"]
                # Stable balanced BCE computed analytically, without RNG or mode changes.
                loss = 0.5 * (max(-positive_logit, 0) + math.log1p(math.exp(-abs(positive_logit)))
                              + max(negative_logit, 0) + math.log1p(math.exp(-abs(negative_logit))))
                self.rows.append({"optimizer_updates": self.updates, "length": length,
                                  "evaluation_kind": "closed_form", **values, **totals,
                                  "alpha": alpha, "effective_margin": alpha * values["delta"],
                                  "target_mass": mass, "positive_logit": positive_logit,
                                  "negative_logit": negative_logit, "balanced_bce": loss})
            write_csv(self.output / "training_diagnostics.csv", self.rows)
        if self.reference_run is not None and self.updates == self.verify_step:
            checkpoint = torch.load(self.reference_run / "model.pt", map_location="cpu", weights_only=False)
            assert checkpoint["optimizer_updates"] == self.verify_step
            current = model.state_dict()
            equal = all(torch.equal(value.detach().cpu(), checkpoint["state_dict"][key])
                        for key, value in current.items())
            assert equal, f"Pilot weights differ at update {self.verify_step}"
            with (self.reference_run / "training_diagnostics.csv").open(newline="", encoding="utf-8") as stream:
                reference_rows = list(csv.DictReader(stream))
            with (self.output / "training_diagnostics.csv").open(newline="", encoding="utf-8") as stream:
                current_rows = list(csv.DictReader(stream))
            assert current_rows == reference_rows, "Pilot diagnostic prefix mismatch"
            save_model_checkpoint(self.output / f"model_step{self.verify_step}.pt", model=model,
                                  config=self.config, optimizer_updates=self.verify_step)
            write_json(self.output / "pilot_prefix_verification.json", {
                "reference_run": str(self.reference_run), "verified_step": self.verify_step,
                "initial_state_equal": True, "weights_bitwise_equal": True,
                "diagnostic_rows_exactly_equal": len(current_rows),
            })


def run_one(config, output, device, interval, reference_run=None, verify_step=1600):
    output.mkdir()
    write_json(output / "config.json", {**asdict(config), "resolved_device": str(device)})
    set_reproducibility(config.seed)
    recorder = TrainingRecorder(config, output, interval, reference_run, verify_step)
    started = time.perf_counter()
    model, updates = train_model(config, device=device, output_dir=output, training_observer=recorder)
    assert updates == recorder.updates == config.max_train_steps
    assert sum(entry["sequences"] for entry in recorder.exposure.values()) == updates * config.batch_size
    save_model_checkpoint(output / "model.pt", model=model, config=config, optimizer_updates=updates)
    config_record = {**asdict(config), "resolved_device": str(device), "optimizer_updates": updates,
                     "initial_state_sha256": recorder.initial_digest,
                     "train_length_count": len(config.train_lengths),
                     "examples_per_train_length": config.train_examples, "final_target_allowed": False}
    write_json(output / "config.json", config_record)
    values = mechanism(model)
    explicit_rows, position_rows, non_target_rows, target_rows = [], [], [], []
    for length in EXPLICIT_LENGTHS:
        row, positions, non_targets, targets = evaluate_length(
            model, length=length, examples=config.test_examples, eval_chunk_examples=config.eval_chunk_examples,
            eval_sampling_mode=config.eval_sampling_mode, batch_size=config.eval_batch_size,
            seed=config.seed + 10000 + length, target_position_mode=config.target_position_mode,
            target_token_count=config.target_token_count, non_target_token_count=config.non_target_token_count,
            non_target_sampling=config.non_target_sampling, device=device,
        )
        gap, _ = attention_gap(config.alpha_mode, values, math.log(length))
        assert math.isclose(row["mean_empirical_target_attention"], sigmoid(gap), rel_tol=3e-5, abs_tol=3e-6)
        row.update(train_lengths=" ".join(map(str, config.train_lengths)), train_length_count=len(config.train_lengths),
                   optimizer_updates=updates, examples_per_train_length=config.train_examples, final_target_allowed=False)
        explicit_rows.append(row)
        position_rows.extend(positions)
        non_target_rows.extend(non_targets)
        target_rows.extend(targets)
    add_train_delta_theory(explicit_rows, train_length=config.train_lengths[0])
    for filename, rows in [("metrics_by_length.csv", explicit_rows), ("target_position_metrics.csv", position_rows),
                           ("non_target_type_metrics.csv", non_target_rows), ("target_type_metrics.csv", target_rows)]:
        if rows:
            write_csv(output / filename, rows)
    predictions = []
    for length in ANALYTICAL_LENGTHS:
        gap, alpha = attention_gap(config.alpha_mode, values, math.log(length))
        mass = sigmoid(gap)
        zpos = values["head_slope"] * mass + values["head_intercept"]
        predictions.append({"length": str(length), "evaluation_kind": "closed_form",
                            "alpha": alpha, "effective_margin": alpha * values["delta"], "target_mass": mass,
                            "positive_logit": zpos, "negative_logit": values["head_intercept"],
                            "positive_prediction": int(zpos >= 0), "negative_prediction": int(values["head_intercept"] >= 0)})
    write_csv(output / "analytical_predictions.csv", predictions)
    exposure = {"by_length": recorder.exposure,
                "total_sequences": sum(e["sequences"] for e in recorder.exposure.values()),
                "total_tokens": sum(e["tokens"] for e in recorder.exposure.values())}
    write_json(output / "exposure.json", exposure)
    with (output / "train_history.csv").open(encoding="utf-8", newline="") as stream:
        history = list(csv.DictReader(stream))
    if reference_run is not None:
        with (reference_run / "train_history.csv").open(newline="", encoding="utf-8") as stream:
            reference_history = list(csv.DictReader(stream))
        assert history[:len(reference_history)] == reference_history, "Pilot training-history prefix mismatch"
        verification = json.loads((output / "pilot_prefix_verification.json").read_text())
        verification["train_history_rows_exactly_equal"] = len(reference_history)
        write_json(output / "pilot_prefix_verification.json", verification)
    return {"run": output.name, "alpha_mode": config.alpha_mode, "seed": config.seed,
            "train_lengths": " ".join(map(str, config.train_lengths)), "optimizer_updates": updates,
            "initial_state_sha256": recorder.initial_digest, **values,
            "total_sequences": exposure["total_sequences"], "total_tokens": exposure["total_tokens"],
            "train_loss": float(history[-1]["train_loss"]), "val_loss": float(history[-1]["val_loss"]),
            "positive_accuracy_10000": explicit_rows[-1]["positive_accuracy"],
            "negative_accuracy_10000": explicit_rows[-1]["negative_accuracy"],
            "target_mass_10000": explicit_rows[-1]["mean_empirical_target_attention"],
            "positive_logit_10000": explicit_rows[-1]["mean_logit_positive"],
            "predicted_failure_log10_length": predicted_failure_log10(config.alpha_mode, values),
            "wall_seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wall-time-minutes", type=float, required=True)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--diagnostic-interval", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    parser.add_argument("--modes", choices=MODES, nargs="+", default=MODES)
    parser.add_argument("--reference-pilot", help="Verify a fresh restart against this completed pilot at --verify-step.")
    parser.add_argument("--verify-step", type=int, default=1600)
    args = parser.parse_args()
    if min(args.steps, args.diagnostic_interval, args.wall_time_minutes) <= 0:
        parser.error("Budgets and diagnostic interval must be positive.")
    if args.reference_pilot and (args.verify_step > args.steps or args.verify_step % args.diagnostic_interval):
        parser.error("Reference verification step must be recorded and not exceed the training budget.")
    reference_pilot = project_dir() / args.reference_pilot if args.reference_pilot else None
    output = project_dir() / args.output_dir
    if output.exists():
        raise FileExistsError(f"Use a new pilot directory: {output}")
    output.mkdir(parents=True)
    device = resolve_device(args.device)
    repo = project_dir().parent
    specs = []
    for mode in args.modes:
        for seed in args.seeds:
            for lengths in LENGTH_SETS:
                name = f"{mode}_n{'_'.join(map(str, lengths))}_s{seed}"
                specs.append(Stage3Config(seed=seed, device=args.device, alpha_mode=mode,
                                         train_length=lengths[0], train_lengths=lengths, train_examples=2048,
                                         val_examples=512, batch_size=64, eval_batch_size=16,
                                         test_examples=50, eval_chunk_examples=50, eval_sampling_mode="random",
                                         eval_lengths=EXPLICIT_LENGTHS, max_train_steps=args.steps,
                                         output_dir=(output / name).relative_to(project_dir()).as_posix()))
    environment = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                   "dirty_status": subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True).strip(),
                   "python": platform.python_version(), "torch": torch.__version__, "device": str(device),
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "cuda_runtime": torch.version.cuda}
    write_json(output / "manifest.json", {"environment": environment, "launch_arguments": vars(args),
                                           "planned_configs": [asdict(config) for config in specs],
                                           "diagnostics": f"initial and every {args.diagnostic_interval} updates; final; verification model checkpoint only; no optimizer checkpoints",
                                           "larger_lengths": "closed-form only; no explicit 10M evaluation"})
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for source in [Path(__file__), project_dir() / "src/stage3_simplified_attention.py",
                   project_dir() / "src/analyze_stage3_mechanism.py",
                   project_dir() / "documents/training_length/PLAN.md"]:
        shutil.copy2(source, snapshot / source.name)
    (snapshot / "working_diff.patch").write_text(subprocess.check_output(["git", "diff"], cwd=repo, text=True), encoding="utf-8")
    started = time.perf_counter()
    rows = []
    pairing = {}
    for config in specs:
        remaining = args.wall_time_minutes * 60 - (time.perf_counter() - started)
        if remaining <= 0 or (rows and remaining < max(row["wall_seconds"] for row in rows) * 1.2):
            break
        name = Path(config.output_dir).name
        print(f"START {len(rows) + 1}/{len(specs)} {name}", flush=True)
        reference_run = reference_pilot / name if reference_pilot is not None else None
        row = run_one(config, output / name, device, args.diagnostic_interval, reference_run, args.verify_step)
        key = (config.alpha_mode, config.seed)
        if key in pairing:
            assert pairing[key] == row["initial_state_sha256"], f"Initialization mismatch: {name}"
        pairing[key] = row["initial_state_sha256"]
        rows.append(row)
        write_csv(output / "pilot_summary.csv", rows)
        write_json(output / "status.json", {"completed_runs": len(rows), "planned_runs": len(specs),
                                            "wall_seconds": time.perf_counter() - started,
                                            "complete": len(rows) == len(specs)})
        print(f"DONE {name} seconds={row['wall_seconds']:.2f} delta={row['delta']:.4f} c_delta={row['c_delta']} acc10k={row['positive_accuracy_10000']}", flush=True)
    if len(rows) != len(specs):
        raise RuntimeError(f"Wall-time cap reached: completed {len(rows)}/{len(specs)}")
    print(f"COMPLETE {len(rows)} runs at {output}", flush=True)


if __name__ == "__main__":
    main()
