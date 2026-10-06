"""Run coefficient controls with behavior matched at a declared initial length."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
import platform
import random
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_training_length_init import softplus_inverse
from run_training_length_pilot import EXPLICIT_LENGTHS, run_one, state_digest
from run_training_length_trace import UpdateTrace, prediction, trace_state, verify_replay
from stage3_simplified_attention import (
    SimplifiedLastQueryAttentionClassifier, Stage3Config, project_dir,
    resolve_device, set_reproducibility, write_csv, write_json,
)

RTOL, ATOL = 3e-5, 3e-6
TRACE_STEPS = set(range(1, 201)) | set(range(1590, 1611)) | set(range(3190, 3211))


def rng_digest():
    states = [random.getstate(), np.random.get_state(), torch.get_rng_state().numpy().tobytes()]
    if torch.cuda.is_available():
        states.extend(state.cpu().numpy().tobytes() for state in torch.cuda.get_rng_state_all())
    return hashlib.sha256(pickle.dumps(states)).hexdigest()


def parameter_norms(model):
    return {name: float(parameter.detach().cpu().double().norm().item()) for name, parameter in
            (("query_norm", model.query_projection.weight), ("key_norm", model.key_projection.weight))}


def balanced_bce(positive, negative):
    return 0.5 * (max(-positive, 0) + math.log1p(math.exp(-abs(positive)))
                  + max(negative, 0) + math.log1p(math.exp(-abs(negative))))


def initial_metrics(model):
    state = trace_state(model)
    values = {key: value for key, value in state.items() if key not in {"q", "kd"}}
    values.update(parameter_norms(model), c_delta=state["c"] * state["delta"])
    for length in EXPLICIT_LENGTHS:
        mass, logit = prediction(state, length)
        values.update({f"target_mass_{length}": mass, f"positive_logit_{length}": logit,
                       f"effective_margin_{length}": (1 + state["c"] * math.log1p(length)) * state["delta"],
                       f"balanced_bce_{length}": balanced_bce(logit, state["head_intercept"])})
    return values


def fixed_inputs(length, device):
    tokens = torch.ones((4, length), dtype=torch.long, device=device)
    for row, position in enumerate((0, (length - 2) // 2, length - 2)):
        tokens[row, position] = 0
    labels = torch.tensor([1.0, 1.0, 1.0, 0.0], device=device)
    return tokens, labels


@torch.no_grad()
def compare_functions(first, second, lengths):
    residuals = {}
    device = next(first.parameters()).device
    for length in lengths:
        tokens, labels = fixed_inputs(length, device)
        a, da = first(tokens, return_details=True)
        b, db = second(tokens, return_details=True)
        for key in ("corrected_scores", "attention_weights", "attention_output"):
            torch.testing.assert_close(da[key], db[key], rtol=RTOL, atol=ATOL)
            residuals[f"{key}_{length}"] = float((da[key] - db[key]).abs().max().item())
        torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL)
        residuals[f"logits_{length}"] = float((a - b).abs().max().item())
        la = F.binary_cross_entropy_with_logits(a, labels, reduction="none")
        lb = F.binary_cross_entropy_with_logits(b, labels, reduction="none")
        torch.testing.assert_close(la, lb, rtol=RTOL, atol=ATOL)
        residuals[f"loss_{length}"] = float((la - lb).abs().max().item())
    return residuals


@torch.no_grad()
def apply_matched_initialization(model, reference_length, factorization="symmetric", output=None,
                                 comparison_state=None):
    """Transform explicitly before optimizer creation; preserve the RNG stream."""
    if factorization not in {"symmetric", "query_only", "key_only"}:
        raise ValueError(factorization)
    rng_before = rng_digest()
    baseline = copy.deepcopy(model)
    baseline.alpha_log_scale_unconstrained.fill_(-5.0)
    baseline_values = initial_metrics(baseline)
    coefficient = model.learned_alpha_coefficient()
    c_base = baseline.learned_alpha_coefficient()
    ratio = (1 + c_base * math.log1p(reference_length)) / (1 + coefficient * math.log1p(reference_length))
    q_scale, k_scale = {
        "symmetric": (math.sqrt(ratio), math.sqrt(ratio)),
        "query_only": (ratio, 1.0), "key_only": (1.0, ratio),
    }[factorization]
    if ratio != 1.0:
        model.query_projection.weight.mul_(q_scale)
        model.key_projection.weight.mul_(k_scale)
    assert torch.equal(model.classifier.weight, baseline.classifier.weight)
    assert torch.equal(model.classifier.bias, baseline.classifier.bias)
    for current, old, scale in ((model.query_projection.weight, baseline.query_projection.weight, q_scale),
                                (model.key_projection.weight, baseline.key_projection.weight, k_scale)):
        torch.testing.assert_close(current, old * scale, rtol=RTOL, atol=ATOL)
        for column in range(current.shape[1]):
            assert float((current[:, column].double() * old[:, column].double()).sum()) >= 0
    bq = baseline.query_projection.weight.detach().cpu().double()[:, 1]
    bk = baseline.key_projection.weight.detach().cpu().double()
    mq = model.query_projection.weight.detach().cpu().double()[:, 1]
    mk = model.key_projection.weight.detach().cpu().double()
    old_scores = bq @ bk / math.sqrt(model.d_head)
    new_scores = mq @ mk / math.sqrt(model.d_head)
    torch.testing.assert_close(new_scores, ratio * old_scores, rtol=RTOL, atol=ATOL)
    residuals = compare_functions(baseline, model, [reference_length])
    factor_residuals = None
    if comparison_state is not None:
        comparison = copy.deepcopy(model)
        comparison.load_state_dict(comparison_state)
        factor_residuals = compare_functions(comparison, model, EXPLICIT_LENGTHS)
        torch.testing.assert_close(comparison.query_projection.weight.double()[:, 1] @
                                   comparison.key_projection.weight.double(),
                                   model.query_projection.weight.double()[:, 1] @ model.key_projection.weight.double(),
                                   rtol=RTOL, atol=ATOL)
    matched_values = initial_metrics(model)
    for length in EXPLICIT_LENGTHS:
        tokens, _ = fixed_inputs(length, next(model.parameters()).device)
        a, b = baseline(tokens), model(tokens)
        torch.testing.assert_close(a[-1], b[-1], rtol=RTOL, atol=ATOL)
    rng_after = rng_digest()
    assert rng_before == rng_after, "Initial transform consumed RNG"
    result = {"reference_length": reference_length, "factorization": factorization,
              "ratio": ratio, "query_scale": q_scale, "key_scale": k_scale,
              "baseline_initial_sha256": state_digest(baseline),
              "matched_initial_sha256": state_digest(model),
              "head_bitwise_equal": True, "rng_unchanged": True,
              "baseline": baseline_values, "matched": matched_values,
              "max_raw_score_scaling_residual": float((new_scores - ratio * old_scores).abs().max()),
              "reference_function_residuals": residuals,
              "factorization_function_residuals": factor_residuals}
    if output is not None:
        torch.save({"state_dict": {key: value.detach().cpu().clone() for key, value in baseline.state_dict().items()}},
                   output / "initial_baseline.pt")
        torch.save({"state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}},
                   output / "initial_matched.pt")
        write_json(output / "initial_matching.json", result)
    return result


class MatchedUpdateTrace(UpdateTrace):
    """Extend exact update accounting with complete projection norms and losses."""

    def __init__(self, output, steps=TRACE_STEPS):
        super().__init__(set(steps), output)
        self.norm_rows = []
        self.parameter_before = None

    def record_norms(self, model, step):
        self.norm_rows.append({"optimizer_updates": step, **initial_metrics(model)})
        write_csv(self.output / "norm_diagnostics.csv", self.norm_rows)

    def before_step(self, model, tokens, labels, loss, optimizer):
        if self.step + 1 in self.steps:
            self.parameter_before = {key: value.detach().cpu().double().clone() for key, value in
                                     (("query", model.query_projection.weight), ("key", model.key_projection.weight))}
        super().before_step(model, tokens, labels, loss, optimizer)

    def after_step(self, model, tokens, labels, loss, optimizer):
        super().after_step(model, tokens, labels, loss, optimizer)
        if self.step in self.steps:
            row = self.rows[-1]
            for key, parameter in (("query", model.query_projection.weight), ("key", model.key_projection.weight)):
                old = self.parameter_before[key]
                current = parameter.detach().cpu().double()
                update = current - old
                row.update({f"{key}_parameter_norm_before": float(old.norm()),
                            f"{key}_parameter_norm_after": float(current.norm()),
                            f"{key}_full_update": json.dumps(update.reshape(-1).tolist()),
                            f"{key}_full_update_norm": float(update.norm()),
                            f"{key}_relative_update_norm": float(update.norm() / old.norm()) if old.norm() > 0 else None})
            for length in (10, 100):
                for timing in ("before", "after"):
                    row[f"balanced_bce_{length}_{timing}"] = balanced_bce(
                        row[f"positive_logit_{length}_{timing}"], row[f"head_intercept_{timing}"])
            if self.step % 200 == 0 or self.step == max(self.steps):
                write_csv(self.output / "update_trace.csv", self.rows)
            self.parameter_before = None
        if self.step % 200 == 0:
            self.record_norms(model, self.step)


def make_config(seed, length, coefficient, output, device):
    return Stage3Config(seed=seed, device=device, alpha_mode="learned_log",
        alpha_log_scale_init=-5.0 if coefficient is None else softplus_inverse(coefficient),
        train_length=length, train_lengths=(length,), train_examples=2048, val_examples=512,
        batch_size=64, eval_batch_size=16, test_examples=50, eval_chunk_examples=50,
        eval_sampling_mode="random", eval_lengths=EXPLICIT_LENGTHS, max_train_steps=6400,
        output_dir=output.relative_to(project_dir()).as_posix())


def make_model(config, device):
    set_reproducibility(config.seed)
    return SimplifiedLastQueryAttentionClassifier(d_head=config.d_head, alpha_mode=config.alpha_mode,
        alpha_log_scale_init=config.alpha_log_scale_init,
        target_token_count=config.target_token_count, non_target_token_count=config.non_target_token_count).to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase", choices=["main", "factorization"], required=True)
    parser.add_argument("--baseline-dir", default="runs/training_length/seed_replication_20260929_01")
    parser.add_argument("--symmetric-dir")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    parser.add_argument("--wall-time-minutes", type=float, default=60)
    parser.add_argument("--prior-training-seconds", type=float, default=0)
    args = parser.parse_args()
    if args.wall_time_minutes <= 0 or args.prior_training_seconds < 0:
        parser.error("Invalid time budget")
    if args.phase == "factorization" and not args.symmetric_dir:
        parser.error("Factorization phase requires --symmetric-dir")
    output = project_dir() / args.output_dir
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    baseline = project_dir() / args.baseline_dir
    symmetric = project_dir() / args.symmetric_dir if args.symmetric_dir else None
    device = resolve_device(args.device)
    specs = []
    if args.phase == "main":
        for seed in range(3, 8):
            for coefficient in (0.1, 1.0):
                tag = "c0p1" if coefficient == 0.1 else "c1"
                for reference, length in ((100, 10), (100, 100), (10, 10)):
                    name = f"n{length}_s{seed}_{tag}_ref{reference}_symmetric"
                    specs.append((make_config(seed, length, coefficient, output / name, args.device),
                                  name, reference, "symmetric", None))
    else:
        for seed in range(3, 6):
            for length in (10, 100):
                comparison = symmetric / f"n{length}_s{seed}_c1_ref{length}_symmetric" / "initial_matched.pt"
                state = torch.load(comparison, map_location=device, weights_only=False)["state_dict"]
                for factor in ("query_only", "key_only"):
                    name = f"n{length}_s{seed}_c1_ref{length}_{factor}"
                    specs.append((make_config(seed, length, 1.0, output / name, args.device),
                                  name, length, factor, state))
    preflight = []
    for config, name, reference, factor, comparison in specs:
        model = make_model(config, device)
        check = apply_matched_initialization(model, reference, factor, comparison_state=comparison)
        old_config = json.loads((baseline / f"learned_log_n{config.train_length}_s{config.seed}" / "config.json").read_text())
        assert check["baseline_initial_sha256"] == old_config["initial_state_sha256"]
        preflight.append({"run": name, "seed": config.seed, "train_length": config.train_length, **check})
    write_json(output / "preflight.json", preflight)
    repo = project_dir().parent
    environment = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True).strip(),
        "python": platform.python_version(), "torch": torch.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_runtime": torch.version.cuda}
    write_json(output / "manifest.json", {"environment": environment, "launch_arguments": vars(args),
        "planned_configs": [asdict(config) for config, _, _, _, _ in specs],
        "comparison_checkpoints": "Initial-state comparisons only; never resumed or loaded to initialize training"})
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for source in (Path(__file__), project_dir() / "src/stage3_simplified_attention.py",
                   project_dir() / "src/run_training_length_pilot.py", project_dir() / "src/run_training_length_trace.py",
                   project_dir() / "src/run_training_length_init.py", project_dir() / "src/analyze_stage3_mechanism.py",
                   project_dir() / "documents/training_length/MATCHED_INITIAL_BEHAVIOR_SPEC.md"):
        shutil.copy2(source, snapshot / source.name)
    (snapshot / "working_diff.patch").write_text(subprocess.check_output(["git", "diff"], cwd=repo, text=True), encoding="utf-8")
    started = time.perf_counter()
    verification_rows, rows = [], []

    def remaining():
        return args.wall_time_minutes * 60 - args.prior_training_seconds - (time.perf_counter() - started)

    if args.phase == "main":
        for seed, length in ((3, 100), (5, 10)):
            assert remaining() > 60, "Budget exhausted before baseline verification"
            name = f"baseline_verification_n{length}_s{seed}"
            config = make_config(seed, length, None, output / name, args.device)
            trace = MatchedUpdateTrace(output / name)
            def initialize(model, reference=length, target=output / name, observer=trace):
                apply_matched_initialization(model, reference, output=target)
                observer.record_norms(model, 0)
            print(f"VERIFY {name}", flush=True)
            row = run_one(config, output / name, device, 200, update_observer=trace,
                          initial_state_transform=initialize)
            old = baseline / f"learned_log_n{length}_s{seed}"
            verified = verify_replay(output / name, old)
            assert state_digest(make_model(config, device)) == row["initial_state_sha256"]
            verified.update(prefix_through_1600_exact=True, verified_budget=6400,
                            reference=str(old), prefix_diagnostic_rows=36, prefix_history_rows=50)
            write_json(output / name / "baseline_verification.json", verified)
            verification_rows.append(row)
            write_csv(output / "baseline_verification_summary.csv", verification_rows)
            print(f"VERIFIED {name} seconds={row['wall_seconds']:.2f}", flush=True)
    for config, name, reference, factor, comparison in specs:
        if remaining() < (max((row["wall_seconds"] for row in rows + verification_rows), default=60) * 1.2):
            break
        trace = MatchedUpdateTrace(output / name)
        def initialize(model, ref=reference, fact=factor, target=output / name, state=comparison, observer=trace):
            apply_matched_initialization(model, ref, fact, target, state)
            observer.record_norms(model, 0)
        print(f"START {len(rows) + 1}/{len(specs)} {name}", flush=True)
        row = run_one(config, output / name, device, 200, update_observer=trace,
                      initial_state_transform=initialize)
        check = json.loads((output / name / "initial_matching.json").read_text())
        assert row["initial_state_sha256"] == check["matched_initial_sha256"]
        assert len(trace.rows) == len(TRACE_STEPS)
        assert max(trace.max_residual.values()) < 1e-10
        row.update(reference_length=reference, factorization=factor,
                   initial_coefficient=check["matched"]["c"], ratio=check["ratio"],
                   traced_updates=len(trace.rows), max_update_residual=max(trace.max_residual.values()),
                   max_initial_matching_residual=max(check["reference_function_residuals"].values()))
        rows.append(row)
        write_csv(output / "summary.csv", rows)
        write_json(output / "status.json", {"phase": args.phase, "completed_runs": len(rows),
                   "planned_runs": len(specs), "verification_runs": len(verification_rows),
                   "wall_seconds": time.perf_counter() - started,
                   "prior_training_seconds": args.prior_training_seconds,
                   "complete": len(rows) == len(specs)})
        print(f"DONE {name} seconds={row['wall_seconds']:.2f} c_delta={row['c_delta']:.6f}", flush=True)
    if len(rows) != len(specs):
        raise RuntimeError(f"Time cap: completed {len(rows)}/{len(specs)}")
    print(f"COMPLETE {args.phase}: {len(rows)} runs", flush=True)


if __name__ == "__main__":
    main()
