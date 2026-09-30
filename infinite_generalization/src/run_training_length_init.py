"""Run paired learned-log coefficient-initialization interventions."""

from __future__ import annotations

import argparse
import hashlib
import math
import platform
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

from run_training_length_pilot import EXPLICIT_LENGTHS, attention_gap, mechanism, run_one, sigmoid, state_digest
from stage3_simplified_attention import (
    SimplifiedLastQueryAttentionClassifier, Stage3Config, project_dir, resolve_device,
    set_reproducibility, write_csv, write_json,
)


def softplus_inverse(coefficient: float) -> float:
    if coefficient <= 0 or not math.isfinite(coefficient):
        raise ValueError("Initial coefficient must be finite and positive.")
    return coefficient + math.log(-math.expm1(-coefficient))


def score_state_digest(model) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key == "alpha_log_scale_unconstrained":
            continue
        digest.update(key.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initial_state(config):
    set_reproducibility(config.seed)
    model = SimplifiedLastQueryAttentionClassifier(
        d_head=config.d_head, alpha_mode=config.alpha_mode,
        alpha_log_scale_init=config.alpha_log_scale_init,
        target_token_count=config.target_token_count,
        non_target_token_count=config.non_target_token_count,
    )
    values = mechanism(model)
    result = {
        "initial_state_sha256": state_digest(model),
        "noncoefficient_state_sha256": score_state_digest(model),
        "u_initial": float(model.alpha_log_scale_unconstrained.item()),
        "c_initial": values["c"], **values,
    }
    for length in (10, 100):
        gap, alpha = attention_gap("learned_log", values, math.log(length))
        result[f"effective_margin_{length}"] = alpha * values["delta"]
        result[f"target_mass_{length}"] = sigmoid(gap)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--wall-time-minutes", type=float, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5, 6, 7])
    parser.add_argument("--initial-coefficients", type=float, nargs="+", default=[0.1, 1.0])
    args = parser.parse_args()
    output = project_dir() / args.output_dir
    baseline = project_dir() / args.baseline_dir
    if output.exists():
        raise FileExistsError(output)
    if not baseline.is_dir():
        raise FileNotFoundError(baseline)
    output.mkdir(parents=True)
    device = resolve_device(args.device)
    repo = project_dir().parent
    specs = []
    preflight = []
    pairing = {}
    for seed in args.seeds:
        for coefficient in (math.log1p(math.exp(-5)), *args.initial_coefficients):
            u = -5.0 if coefficient < 0.01 else softplus_inverse(coefficient)
            tag = "baseline" if coefficient < 0.01 else f"c{coefficient:g}".replace(".", "p")
            for lengths in ((10,), (100,)):
                name = f"learned_log_n{lengths[0]}_s{seed}_{tag}"
                config = Stage3Config(
                    seed=seed, device=args.device, alpha_mode="learned_log",
                    alpha_log_scale_init=u, train_length=lengths[0], train_lengths=lengths,
                    train_examples=2048, val_examples=512, batch_size=64,
                    eval_batch_size=16, test_examples=50, eval_chunk_examples=50,
                    eval_sampling_mode="random", eval_lengths=EXPLICIT_LENGTHS,
                    max_train_steps=6400,
                    output_dir=(output / name).relative_to(project_dir()).as_posix(),
                )
                initial = initial_state(config)
                previous = pairing.setdefault(seed, initial["noncoefficient_state_sha256"])
                assert previous == initial["noncoefficient_state_sha256"], f"Noncoefficient pairing failed: {name}"
                if tag == "baseline":
                    reference = baseline / f"learned_log_n{lengths[0]}_s{seed}"
                    from json import loads
                    reference_config = loads((reference / "config.json").read_text())
                    assert initial["initial_state_sha256"] == reference_config["initial_state_sha256"]
                preflight.append({"run": name, "seed": seed, "train_lengths": " ".join(map(str, lengths)),
                                  "requested_initial_c": coefficient, **initial})
                if tag != "baseline":
                    specs.append((config, tag, coefficient))
    write_csv(output / "initial_states.csv", preflight)
    environment = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                   "dirty_status": subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True).strip(),
                   "python": platform.python_version(), "torch": torch.__version__, "device": str(device),
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "cuda_runtime": torch.version.cuda}
    write_json(output / "manifest.json", {"environment": environment, "launch_arguments": vars(args),
                                          "planned_configs": [asdict(config) for config, _, _ in specs],
                                          "baseline_reused": str(baseline)})
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for source in [Path(__file__), project_dir() / "src/run_training_length_pilot.py",
                   project_dir() / "src/stage3_simplified_attention.py",
                   project_dir() / "src/analyze_stage3_mechanism.py",
                   project_dir() / "documents/training_length/MECHANISM_FOLLOWUP_SPEC.md"]:
        shutil.copy2(source, snapshot / source.name)
    (snapshot / "working_diff.patch").write_text(subprocess.check_output(["git", "diff"], cwd=repo, text=True), encoding="utf-8")
    started = time.perf_counter()
    rows = []
    for config, tag, coefficient in specs:
        remaining = args.wall_time_minutes * 60 - (time.perf_counter() - started)
        if remaining <= 0 or (rows and remaining < max(row["wall_seconds"] for row in rows) * 1.2):
            break
        name = Path(config.output_dir).name
        print(f"START {len(rows) + 1}/{len(specs)} {name}", flush=True)
        row = run_one(config, output / name, device, interval=200)
        assert row["initial_state_sha256"] == next(item["initial_state_sha256"] for item in preflight if item["run"] == name)
        row.update(initial_coefficient_requested=coefficient, initial_tag=tag)
        rows.append(row)
        write_csv(output / "intervention_summary.csv", rows)
        write_json(output / "status.json", {"completed_runs": len(rows), "planned_runs": len(specs),
                                                "wall_seconds": time.perf_counter() - started,
                                                "complete": len(rows) == len(specs)})
        print(f"DONE {name} seconds={row['wall_seconds']:.2f} c_delta={row['c_delta']:.6f}", flush=True)
    if len(rows) != len(specs):
        raise RuntimeError(f"Wall-time cap reached: completed {len(rows)}/{len(specs)}")
    print(f"COMPLETE {len(rows)} runs at {output}", flush=True)


if __name__ == "__main__":
    main()
