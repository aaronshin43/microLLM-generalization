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

                logits, pooled, attention_layers = model.forward_with_attention(batch_tokens)
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
                for summary in summarize_attention_for_sample(
                    attention_layers=attention_layers,
                    tokens=tokens,
                    target_token=task.target_token,
                ):
                    rows.append({**base_row, **summary})

    write_attention_summary_csv(attention_dir / "attention_summary.csv", rows)
