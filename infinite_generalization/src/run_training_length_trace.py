"""Replay paired runs and record realized learned-log optimizer updates."""

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

import torch

from run_training_length_pilot import run_one
from stage3_simplified_attention import Stage3Config, project_dir, resolve_device, write_csv, write_json


def norm(values):
    return math.sqrt(sum(value * value for value in values))


def trace_state(model):
    query = model.query_projection.weight.detach().cpu().to(torch.float64)
    key = model.key_projection.weight.detach().cpu().to(torch.float64)
    q = query[:, 1].tolist()
    kd = (key[:, 0] - key[:, 1]).tolist()
    coefficient = model.learned_alpha_coefficient()
    raw_gap = sum(x * y for x, y in zip(q, kd)) / math.sqrt(model.d_head)
    head = model.classifier.weight.detach().cpu()[0].tolist()
    intercept = float(head[1] + model.classifier.bias.item())
    slope = float(head[0] - head[1])
    return {"q": q, "kd": kd, "u": float(model.alpha_log_scale_unconstrained.item()),
            "c": coefficient, "delta": raw_gap, "head_slope": slope,
            "head_intercept": intercept,
            "head_threshold": -intercept / slope if slope != 0 else None}


def prediction(state, length):
    alpha = 1 + state["c"] * math.log1p(length)
    gap = alpha * state["delta"] - math.log(length - 1)
    mass = 1 / (1 + math.exp(-gap)) if gap >= 0 else math.exp(gap) / (1 + math.exp(gap))
    return mass, state["head_slope"] * mass + state["head_intercept"]


def needed_steps(reference_run):
    with (reference_run / "training_diagnostics.csv").open(newline="", encoding="utf-8") as stream:
        trajectory = [row for row in csv.DictReader(stream) if row["length"] == "100"]
    first = next((int(row["optimizer_updates"]) for row in trajectory if float(row["c_delta"]) > 1), None)
    windows = [(1, 200), (1590, 1610), (3190, 3210)]
    if first is not None:
        windows.append((max(1, first - 10), min(6400, first + 10)))
    return {step for lo, hi in windows for step in range(lo, hi + 1)}, first


class UpdateTrace:
    """Capture detached gradients and weights around each selected AdamW step."""

    def __init__(self, steps, output):
        self.steps, self.output = steps, output
        self.step = 0
        self.before = None
        self.rows = []
        self.max_residual = {"delta": 0.0, "c_delta": 0.0, "effective_margin_10": 0.0,
                             "effective_margin_100": 0.0}

    def before_step(self, model, tokens, labels, loss, optimizer):
        self.step += 1
        if self.step not in self.steps:
            return
        grads = {}
        for key, parameter in [("u", model.alpha_log_scale_unconstrained),
                               ("query", model.query_projection.weight),
                               ("key", model.key_projection.weight),
                               ("head_weight", model.classifier.weight),
                               ("head_bias", model.classifier.bias)]:
            grads[key] = parameter.grad.detach().cpu().to(torch.float64).reshape(-1).tolist()
        self.before = {"state": trace_state(model), "grads": grads,
                       "loss": float(loss.detach().item()), "batch_length": tokens.shape[1],
                       "batch_size": tokens.shape[0], "positives": int(labels.sum().item())}

    def after_step(self, model, tokens, labels, loss, optimizer):
        if self.step not in self.steps:
            return
        assert self.before is not None
        before = self.before["state"]
        after = trace_state(model)
        dq = [b - a for a, b in zip(before["q"], after["q"])]
        dk = [b - a for a, b in zip(before["kd"], after["kd"])]
        qbar = [(a + b) / 2 for a, b in zip(before["q"], after["q"])]
        kbar = [(a + b) / 2 for a, b in zip(before["kd"], after["kd"])]
        query_gap = sum(x * y for x, y in zip(dq, kbar)) / math.sqrt(len(dq))
        key_gap = sum(x * y for x, y in zip(qbar, dk)) / math.sqrt(len(dq))
        dd = after["delta"] - before["delta"]
        dc = after["c"] - before["c"]
        cbar = (before["c"] + after["c"]) / 2
        dbar = (before["delta"] + after["delta"]) / 2
        delta_exponent = after["c"] * after["delta"] - before["c"] * before["delta"]
        c_part_exponent = dbar * dc
        score_part_exponent = cbar * dd
        row = {"optimizer_updates": self.step, "batch_length": self.before["batch_length"],
               "batch_size": self.before["batch_size"], "positives": self.before["positives"],
               "negatives": self.before["batch_size"] - self.before["positives"],
               "batch_loss": self.before["loss"],
               "u_before": before["u"], "u_after": after["u"],
               "u_gradient": self.before["grads"]["u"][0], "u_update": after["u"] - before["u"],
               "c_before": before["c"], "c_after": after["c"], "c_update": dc,
               "delta_before": before["delta"], "delta_after": after["delta"], "delta_update": dd,
               "q_before": json.dumps(before["q"]), "q_after": json.dumps(after["q"]),
               "kd_before": json.dumps(before["kd"]), "kd_after": json.dumps(after["kd"]),
               "q_gradient": json.dumps(self.before["grads"]["query"]),
               "key_gradient": json.dumps(self.before["grads"]["key"]),
               "head_weight_gradient": json.dumps(self.before["grads"]["head_weight"]),
               "head_bias_gradient": self.before["grads"]["head_bias"][0],
               "q_update": json.dumps(dq), "kd_update": json.dumps(dk),
               "q_update_norm": norm(dq), "kd_update_norm": norm(dk),
               "q_gradient_norm": norm(self.before["grads"]["query"]),
               "key_gradient_norm": norm(self.before["grads"]["key"]),
               "delta_from_query_update": query_gap, "delta_from_key_update": key_gap,
               "delta_decomposition_residual": dd - query_gap - key_gap,
               "c_delta_before": before["c"] * before["delta"],
               "c_delta_after": after["c"] * after["delta"],
               "c_delta_update": delta_exponent,
               "c_delta_from_c_update": c_part_exponent,
               "c_delta_from_score_update": score_part_exponent,
               "c_delta_decomposition_residual": delta_exponent - c_part_exponent - score_part_exponent,
               "head_slope_before": before["head_slope"], "head_slope_after": after["head_slope"],
               "head_intercept_before": before["head_intercept"],
               "head_intercept_after": after["head_intercept"],
               "head_threshold_before": before["head_threshold"],
               "head_threshold_after": after["head_threshold"]}
        for length in (10, 100):
            l = math.log1p(length)
            effective_before = (1 + before["c"] * l) * before["delta"]
            effective_after = (1 + after["c"] * l) * after["delta"]
            c_part = l * dbar * dc
            score_part = (1 + cbar * l) * dd
            mass_before, logit_before = prediction(before, length)
            mass_after, logit_after = prediction(after, length)
            row.update({f"effective_margin_{length}_before": effective_before,
                        f"effective_margin_{length}_after": effective_after,
                        f"effective_margin_{length}_update": effective_after - effective_before,
                        f"effective_margin_{length}_from_c_update": c_part,
                        f"effective_margin_{length}_from_score_update": score_part,
                        f"effective_margin_{length}_decomposition_residual": effective_after - effective_before - c_part - score_part,
                        f"target_mass_{length}_before": mass_before,
                        f"target_mass_{length}_after": mass_after,
                        f"positive_logit_{length}_before": logit_before,
                        f"positive_logit_{length}_after": logit_after})
        self.max_residual["delta"] = max(self.max_residual["delta"], abs(row["delta_decomposition_residual"]))
        self.max_residual["c_delta"] = max(self.max_residual["c_delta"], abs(row["c_delta_decomposition_residual"]))
        for length in (10, 100):
            name = f"effective_margin_{length}"
            self.max_residual[name] = max(self.max_residual[name], abs(row[f"{name}_decomposition_residual"]))
        self.rows.append(row)
        if self.step % 200 == 0 or self.step == max(self.steps):
            write_csv(self.output / "update_trace.csv", self.rows)
        self.before = None


def verify_replay(output, reference):
    current = torch.load(output / "model.pt", map_location="cpu", weights_only=False)
    old = torch.load(reference / "model.pt", map_location="cpu", weights_only=False)
    assert all(torch.equal(value, old["state_dict"][key]) for key, value in current["state_dict"].items())
    for filename in ("training_diagnostics.csv", "train_history.csv"):
        with (output / filename).open(newline="", encoding="utf-8") as stream:
            current_rows = list(csv.DictReader(stream))
        with (reference / filename).open(newline="", encoding="utf-8") as stream:
            old_rows = list(csv.DictReader(stream))
        assert current_rows == old_rows, f"Replay changed {filename}"
    return {"weights_bitwise_equal": True, "diagnostics_exact": True, "history_exact": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--intervention-dir", required=True)
    parser.add_argument("--wall-time-minutes", type=float, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--initial-tags", choices=["baseline", "c0p1", "c1"], nargs="+", default=["baseline", "c0p1"])
    parser.add_argument("--lengths", type=int, choices=[10, 100], nargs="+", default=[10, 100])
    args = parser.parse_args()
    output = project_dir() / args.output_dir
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    baseline = project_dir() / args.baseline_dir
    intervention = project_dir() / args.intervention_dir
    device = resolve_device(args.device)
    repo = project_dir().parent
    specs = []
    for seed in args.seeds:
        for tag in args.initial_tags:
            for length in args.lengths:
                old_name = f"learned_log_n{length}_s{seed}" + ("" if tag == "baseline" else f"_{tag}")
                reference = (baseline if tag == "baseline" else intervention) / old_name
                source_config = json.loads((reference / "config.json").read_text())
                name = f"learned_log_n{length}_s{seed}_{tag}"
                config = Stage3Config(**{key: value for key, value in source_config.items() if key in Stage3Config.__dataclass_fields__})
                config = replace(config, output_dir=(output / name).relative_to(project_dir()).as_posix())
                specs.append((config, name, reference))
    write_json(output / "manifest.json", {"environment": {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                    "dirty_status": subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True).strip(),
                    "python": platform.python_version(), "torch": torch.__version__, "device": str(device),
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
                   "launch_arguments": vars(args), "planned_configs": [asdict(config) for config, _, _ in specs]})
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for source in [Path(__file__), project_dir() / "src/run_training_length_pilot.py",
                   project_dir() / "src/stage3_simplified_attention.py",
                   project_dir() / "documents/training_length/MECHANISM_FOLLOWUP_SPEC.md"]:
        shutil.copy2(source, snapshot / source.name)
    (snapshot / "working_diff.patch").write_text(subprocess.check_output(["git", "diff"], cwd=repo, text=True), encoding="utf-8")
    started = time.perf_counter()
    rows = []
    for config, name, reference in specs:
        remaining = args.wall_time_minutes * 60 - (time.perf_counter() - started)
        if remaining <= 0 or (rows and remaining < max(row["wall_seconds"] for row in rows) * 1.2):
            break
        steps, first_crossing = needed_steps(reference)
        trace = UpdateTrace(steps, output / name)
        print(f"START {len(rows) + 1}/{len(specs)} {name}", flush=True)
        summary = run_one(config, output / name, device, interval=200, update_observer=trace)
        assert len(trace.rows) == len(steps)
        assert max(trace.max_residual.values()) < 1e-10, trace.max_residual
        verification = verify_replay(output / name, reference)
        write_json(output / name / "trace_verification.json", {"reference": str(reference),
                   "traced_updates": len(trace.rows), "first_recorded_crossing": first_crossing,
                   "max_decomposition_residuals": trace.max_residual, **verification})
        summary.update(traced_updates=len(trace.rows), first_recorded_crossing=first_crossing,
                       max_decomposition_residual=max(trace.max_residual.values()))
        rows.append(summary)
        write_csv(output / "trace_summary.csv", rows)
        write_json(output / "status.json", {"completed_runs": len(rows), "planned_runs": len(specs),
                   "wall_seconds": time.perf_counter() - started, "complete": len(rows) == len(specs)})
        print(f"DONE {name} seconds={summary['wall_seconds']:.2f} traced={len(trace.rows)}", flush=True)
    if len(rows) != len(specs):
        raise RuntimeError(f"Wall-time cap reached: completed {len(rows)}/{len(specs)}")
    print(f"COMPLETE {len(rows)} runs at {output}", flush=True)


if __name__ == "__main__":
    main()
