"""Attention export utilities for transformer experiments."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn

from audit import select_audit_indices, target_positions
from config import Stage1Config, Stage2AConfig, Stage2BConfig, TaskConfig
from data import (
    diagnostic_slice_specs,
    make_negative_dataset,
    make_positive_dataset,
)


def attention_entropy(attention: torch.Tensor) -> torch.Tensor:
    """Compute entropy over attention source positions.

    The input shape is `(heads, query_length, key_length)`.
    """

    safe_attention = attention.clamp_min(1e-12)
    return -(safe_attention * safe_attention.log()).sum(dim=-1)


def summarize_attention_for_sample(
    *,
    attention_layers: list[torch.Tensor],
    tokens: torch.Tensor,
    target_token: int,
) -> list[dict[str, float | int]]:
    """Summarize attention entropy and target-token attention mass per layer and head."""

    target_mask = tokens.eq(target_token)
    summaries: list[dict[str, float | int]] = []

    for layer_index, attention in enumerate(attention_layers):
        # Remove the batch dimension. Attention shape becomes `(heads, query, key)`.
        sample_attention = attention.squeeze(0).cpu()
        entropy = attention_entropy(sample_attention)

        for head_index in range(sample_attention.shape[0]):
            head_attention = sample_attention[head_index]
            head_entropy = entropy[head_index]

            row: dict[str, float | int] = {
                "layer": layer_index,
                "head": head_index,
                "attention_entropy_mean": float(head_entropy.mean().item()),
                "attention_entropy_max": float(head_entropy.max().item()),
            }

            if target_mask.any():
                target_attention = head_attention[:, target_mask]
                row["attention_to_target_mean"] = float(
                    target_attention.sum(dim=-1).mean().item()
                )
                row["attention_to_target_max"] = float(
                    target_attention.sum(dim=-1).max().item()
                )
            else:
                row["attention_to_target_mean"] = float("nan")
                row["attention_to_target_max"] = float("nan")

            summaries.append(row)

    return summaries


def _tensor_scalar(value: object) -> float:
    """Convert a scalar tensor-like diagnostic value to float."""

    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def summarize_attention_diagnostics_for_sample(
    *,
    attention_details: list[dict[str, object]],
    tokens: torch.Tensor,
    target_token: int,
) -> list[dict[str, float | int | str]]:
    """Summarize length-aware attention score diagnostics per layer and head."""

    target_mask = tokens.eq(target_token).cpu()
    non_target_mask = ~target_mask
    rows: list[dict[str, float | int | str]] = []

    for layer_index, details in enumerate(attention_details):
        base_scores = details["base_scores"].squeeze(0).detach().cpu()
        corrected_scores = details["corrected_scores"].squeeze(0).detach().cpu()
        target_like = details.get("target_like")
        target_like_cpu = (
            target_like.squeeze(0).detach().cpu() if isinstance(target_like, torch.Tensor) else None
        )

        for head_index in range(base_scores.shape[0]):
            head_base = base_scores[head_index]
            head_corrected = corrected_scores[head_index]
            row: dict[str, float | int | str] = {
                "layer": layer_index,
                "head": head_index,
                "attention_variant": str(details.get("attention_variant", "")),
                "length_correction": _tensor_scalar(details["length_correction"]),
                "base_score_mean": float(head_base.mean().item()),
                "base_score_max": float(head_base.max().item()),
                "corrected_score_mean": float(head_corrected.mean().item()),
                "corrected_score_max": float(head_corrected.max().item()),
            }
            if "alpha_length_scale" in details:
                row["alpha_length_scale"] = _tensor_scalar(details["alpha_length_scale"])
            if "beta_length_scale" in details:
                row["beta_length_scale"] = _tensor_scalar(details["beta_length_scale"])

            if target_mask.any():
                target_base = head_base[:, target_mask]
                target_corrected = head_corrected[:, target_mask]
                target_corrected_best = target_corrected.max(dim=-1).values
                non_target_corrected = head_corrected[:, non_target_mask]
                ranks = 1 + non_target_corrected.gt(target_corrected_best.unsqueeze(-1)).sum(dim=-1)
                row["target_base_score_mean"] = float(target_base.mean().item())
                row["target_base_score_max"] = float(target_base.max().item())
                row["target_corrected_score_mean"] = float(target_corrected.mean().item())
                row["target_corrected_score_max"] = float(target_corrected.max().item())
                row["target_corrected_score_rank_mean"] = float(ranks.float().mean().item())
            else:
                row["target_base_score_mean"] = float("nan")
                row["target_base_score_max"] = float("nan")
                row["target_corrected_score_mean"] = float("nan")
                row["target_corrected_score_max"] = float("nan")
                row["target_corrected_score_rank_mean"] = float("nan")

            if target_like_cpu is not None:
                if target_mask.any():
                    row["target_like_target_mean"] = float(target_like_cpu[target_mask].mean().item())
                else:
                    row["target_like_target_mean"] = float("nan")
                if non_target_mask.any():
                    row["target_like_non_target_mean"] = float(
                        target_like_cpu[non_target_mask].mean().item()
                    )
                    row["target_like_non_target_max"] = float(
                        target_like_cpu[non_target_mask].max().item()
                    )
                else:
                    row["target_like_non_target_mean"] = float("nan")
                    row["target_like_non_target_max"] = float("nan")

            rows.append(row)

    return rows


def pooled_activation_stats(pooled: torch.Tensor) -> dict[str, float]:
    """Compute compact diagnostics for a single pooled activation vector."""

    pooled_cpu = pooled.squeeze(0).detach().cpu()
    return {
        "pooled_l2_norm": float(torch.linalg.vector_norm(pooled_cpu, ord=2).item()),
        "pooled_abs_mean": float(pooled_cpu.abs().mean().item()),
        "pooled_dim_min": float(pooled_cpu.min().item()),
        "pooled_dim_max": float(pooled_cpu.max().item()),
    }


def write_attention_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write attention summary rows to CSV."""

    fieldnames = [
        "length",
        "slice",
        "label_type",
        "dataset_index",
        "label",
        "prediction",
        "correct",
        "logit",
        "probability",
        "target_count",
        "target_region",
        "target_positions",
        "pooled_l2_norm",
        "pooled_abs_mean",
        "pooled_dim_min",
        "pooled_dim_max",
        "layer",
        "head",
        "attention_entropy_mean",
        "attention_entropy_max",
        "attention_to_target_mean",
        "attention_to_target_max",
        "attention_variant",
        "length_correction",
        "alpha_length_scale",
        "beta_length_scale",
        "base_score_mean",
        "base_score_max",
        "corrected_score_mean",
        "corrected_score_max",
        "target_base_score_mean",
        "target_base_score_max",
        "target_corrected_score_mean",
        "target_corrected_score_max",
        "target_corrected_score_rank_mean",
        "target_like_target_mean",
        "target_like_non_target_mean",
        "target_like_non_target_max",
        "raw_attention_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def save_attention_analysis(
    model: nn.Module,
    *,
    task: TaskConfig,
    config: Stage1Config | Stage2AConfig | Stage2BConfig,
    device: torch.device,
    output_dir: Path,
    examples_per_class: int,
    save_raw: bool,
) -> None:
    """Save attention summaries for controlled diagnostic slices."""

    if not hasattr(model, "forward_with_attention"):
        raise TypeError("model must implement forward_with_attention() to save attention analysis")

    attention_dir = output_dir / "attention"
    raw_dir = attention_dir / "raw"
    attention_dir.mkdir(parents=True, exist_ok=True)
    if save_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    model.eval()

    for length_index, length in enumerate(task.eval_lengths):
        for slice_index, (slice_name, label_type, target_count, target_region) in enumerate(
            diagnostic_slice_specs(length)
        ):
            generator = torch.Generator().manual_seed(
                config.seed + 40_000 + length_index * 100 + slice_index
            )
            if label_type == "negative":
                inputs, labels = make_negative_dataset(
                    num_examples=config.diagnostic_examples,
                    length=length,
                    task=task,
                    generator=generator,
                )
            else:
                inputs, labels = make_positive_dataset(
                    num_examples=config.diagnostic_examples,
                    length=length,
                    task=task,
                    generator=generator,
                    target_count=target_count,
                    target_region=target_region,
                )

            selected_indices = select_audit_indices(
                labels,
                examples_per_class=examples_per_class,
            )

            for selected_index in selected_indices:
                tokens = inputs[selected_index]
                label = int(labels[selected_index].item())
                batch_tokens = tokens.unsqueeze(0).to(device)

                if hasattr(model, "forward_with_attention_details"):
                    logits, pooled, attention_layers, attention_details = (
                        model.forward_with_attention_details(batch_tokens)
                    )
                else:
                    logits, pooled, attention_layers = model.forward_with_attention(batch_tokens)
                    attention_details = []
                logit = float(logits.squeeze(0).cpu().item())
                probability = float(torch.sigmoid(logits).squeeze(0).cpu().item())
                prediction = int(logit >= 0.0)

                raw_attention_file = ""
                if save_raw:
                    raw_attention_file = (
                        f"length_{length}_{slice_name}_index_{selected_index}_attention.pt"
                    )
                    torch.save(
                        {
                            "tokens": tokens.cpu(),
                            "slice": slice_name,
                            "label_type": label_type,
                            "label": label,
                            "logit": logit,
                            "probability": probability,
                            "pooled": pooled.squeeze(0).cpu(),
                            "attention_layers": [layer.cpu() for layer in attention_layers],
                            "attention_details": [
                                {
                                    key: value.cpu() if isinstance(value, torch.Tensor) else value
                                    for key, value in details.items()
                                }
                                for details in attention_details
                            ],
                        },
                        raw_dir / raw_attention_file,
                    )

                base_row = {
                    "length": length,
                    "slice": slice_name,
                    "label_type": label_type,
                    "dataset_index": selected_index,
                    "label": label,
                    "prediction": prediction,
                    "correct": prediction == label,
                    "logit": logit,
                    "probability": probability,
                    "target_count": int(tokens.eq(task.target_token).sum().item()),
                    "target_region": target_region,
                    "target_positions": target_positions(tokens, target_token=task.target_token),
                    **pooled_activation_stats(pooled),
                    "raw_attention_file": raw_attention_file,
                }
                attention_summaries = summarize_attention_for_sample(
                    attention_layers=attention_layers,
                    tokens=tokens,
                    target_token=task.target_token,
                )
                diagnostic_summaries = summarize_attention_diagnostics_for_sample(
                    attention_details=attention_details,
                    tokens=tokens,
                    target_token=task.target_token,
                )
                diagnostics_by_layer_head = {
                    (summary["layer"], summary["head"]): summary
                    for summary in diagnostic_summaries
                }

                for summary in attention_summaries:
                    key = (summary["layer"], summary["head"])
                    rows.append(
                        {
                            **base_row,
                            **summary,
                            **diagnostics_by_layer_head.get(key, {}),
                        }
                    )

    write_attention_summary_csv(attention_dir / "attention_summary.csv", rows)
