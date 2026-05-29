"""Numerical analysis helpers for transformer length-generalization failures."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from config import Stage1Config, TaskConfig
from data import make_negative_dataset, make_positive_dataset
from models import TransformerTokenPresenceClassifier


def dataclass_values_from_metadata(config_class: type, values: dict[str, Any]) -> dict[str, Any]:
    """Filter old run metadata down to fields accepted by the current dataclass."""

    valid_names = {field.name for field in fields(config_class)}
    return {key: value for key, value in values.items() if key in valid_names}


def load_stage1_checkpoint(run_dir: Path, *, device: torch.device) -> tuple[
    TransformerTokenPresenceClassifier,
    TaskConfig,
    Stage1Config,
    dict[str, Any],
]:
    """Load a Stage 1 checkpoint and its config metadata."""

    metadata = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    task = TaskConfig(**dataclass_values_from_metadata(TaskConfig, metadata["task"]))
    config = Stage1Config(**dataclass_values_from_metadata(Stage1Config, metadata["stage1"]))

    model = TransformerTokenPresenceClassifier(
        vocab_size=task.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    ).to(device)
    state_dict = torch.load(run_dir / "model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, task, config, metadata


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to CSV with fieldnames inferred from the first row."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parameter_stat_rows(model: nn.Module) -> list[dict[str, object]]:
    """Return numerical summaries for trainable parameters."""

    rows: list[dict[str, object]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        values = parameter.detach().float().cpu()
        rows.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "count": parameter.numel(),
                "l2_norm": float(torch.linalg.vector_norm(values).item()),
                "mean": float(values.mean().item()),
                "std": float(values.std(unbiased=False).item()),
                "min": float(values.min().item()),
                "max": float(values.max().item()),
            }
        )
    return rows


def split_qkv(layer: nn.Module) -> dict[str, torch.Tensor]:
    """Split PyTorch MultiheadAttention input projection weights and biases."""

    weight = layer.self_attn.in_proj_weight.detach()
    bias = layer.self_attn.in_proj_bias.detach()
    w_q, w_k, w_v = weight.chunk(3, dim=0)
    b_q, b_k, b_v = bias.chunk(3, dim=0)
    return {
        "w_q": w_q,
        "w_k": w_k,
        "w_v": w_v,
        "b_q": b_q,
        "b_k": b_k,
        "b_v": b_v,
    }


def project_token_embeddings(
    model: TransformerTokenPresenceClassifier,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project vocabulary embeddings into Q, K, and V spaces for layer 0."""

    layer = model.layers[0]
    qkv = split_qkv(layer)
    embeddings = model.embedding.weight.detach()
    q = F.linear(embeddings, qkv["w_q"], qkv["b_q"])
    k = F.linear(embeddings, qkv["w_k"], qkv["b_k"])
    v = F.linear(embeddings, qkv["w_v"], qkv["b_v"])
    return q, k, v


def safe_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity with a stable denominator."""

    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if denominator.item() == 0.0:
        return float("nan")
    return float(torch.dot(a, b).div(denominator).item())


def token_qkv_geometry_rows(
    model: TransformerTokenPresenceClassifier,
    *,
    task: TaskConfig,
) -> list[dict[str, object]]:
    """Summarize target and non-target token geometry in Q/K/V spaces."""

    q, k, v = project_token_embeddings(model)
    target = task.target_token
    d_head = model.layers[0].self_attn.head_dim
    target_q = q[target]
    target_k = k[target]
    target_v = v[target]

    rows: list[dict[str, object]] = []
    for token_id in range(task.vocab_size):
        rows.append(
            {
                "token_id": token_id,
                "is_target": token_id == target,
                "q_norm": float(torch.linalg.vector_norm(q[token_id]).item()),
                "k_norm": float(torch.linalg.vector_norm(k[token_id]).item()),
                "v_norm": float(torch.linalg.vector_norm(v[token_id]).item()),
                "q_cosine_to_target": safe_cosine(q[token_id], target_q),
                "k_cosine_to_target": safe_cosine(k[token_id], target_k),
                "v_cosine_to_target": safe_cosine(v[token_id], target_v),
                "query_to_target_key_score": float(
                    torch.dot(q[token_id], target_k).div(math.sqrt(d_head)).item()
                ),
                "target_query_to_key_score": float(
                    torch.dot(target_q, k[token_id]).div(math.sqrt(d_head)).item()
                ),
            }
        )
    return rows


def make_controlled_sequence(
    *,
    length: int,
    label_type: str,
    task: TaskConfig,
    seed: int,
) -> tuple[torch.Tensor, int | None]:
    """Create one deterministic controlled sequence for numerical analysis."""

    generator = torch.Generator().manual_seed(seed)
    if label_type == "positive":
        inputs, _ = make_positive_dataset(
            num_examples=1,
            length=length,
            task=task,
            generator=generator,
            target_count=1,
            target_region="middle",
        )
        target_index = int(torch.nonzero(inputs[0].eq(task.target_token), as_tuple=False)[0].item())
        return inputs[0], target_index

    if label_type == "negative":
        inputs, _ = make_negative_dataset(
            num_examples=1,
            length=length,
            task=task,
            generator=generator,
        )
        return inputs[0], None

    raise ValueError(f"unknown label_type: {label_type}")


@torch.no_grad()
def manual_layer0_forward(
    model: TransformerTokenPresenceClassifier,
    tokens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run layer 0 manually and return intermediate tensors for analysis."""

    if len(model.layers) != 1:
        raise ValueError("manual analysis currently expects exactly one transformer layer")

    device = next(model.parameters()).device
    tokens = tokens.to(device)
    layer = model.layers[0]
    hidden = model.embedding(tokens.unsqueeze(0)).squeeze(0)
    qkv = split_qkv(layer)

    q_full = F.linear(hidden, qkv["w_q"], qkv["b_q"])
    k_full = F.linear(hidden, qkv["w_k"], qkv["b_k"])
    v_full = F.linear(hidden, qkv["w_v"], qkv["b_v"])

    num_heads = layer.self_attn.num_heads
    d_head = layer.self_attn.head_dim
    length = hidden.shape[0]

    q = q_full.view(length, num_heads, d_head).transpose(0, 1)
    k = k_full.view(length, num_heads, d_head).transpose(0, 1)
    v = v_full.view(length, num_heads, d_head).transpose(0, 1)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head)
    attention = torch.softmax(scores, dim=-1)
    attention_pre_out = torch.matmul(attention, v).transpose(0, 1).reshape(length, -1)
    attention_out = layer.self_attn.out_proj(attention_pre_out)
    after_attention = layer.norm1(hidden + attention_out)
    feedforward = layer.linear2(layer.dropout(layer.activation(layer.linear1(after_attention))))
    encoded = layer.norm2(after_attention + feedforward)
    pooled, argmax_positions = encoded.max(dim=0)
    logit = model.classifier(pooled).squeeze()

    return {
        "embedding": hidden,
        "q": q,
        "k": k,
        "v": v,
        "scores": scores,
        "attention": attention,
        "attention_pre_out": attention_pre_out,
        "attention_out": attention_out,
        "after_attention": after_attention,
        "feedforward": feedforward,
        "encoded": encoded,
        "pooled": pooled,
        "argmax_positions": argmax_positions,
        "logit": logit,
    }


def entropy(attention: torch.Tensor) -> torch.Tensor:
    """Compute entropy over key positions."""

    safe_attention = attention.clamp_min(1e-12)
    return -(safe_attention * safe_attention.log()).sum(dim=-1)


def attention_length_rows(
    model: TransformerTokenPresenceClassifier,
    *,
    task: TaskConfig,
    lengths: list[int],
) -> list[dict[str, object]]:
    """Summarize attention logits, softmax denominators, and target mass by length."""

    rows: list[dict[str, object]] = []
    for length_index, length in enumerate(lengths):
        tokens, target_index = make_controlled_sequence(
            length=length,
            label_type="positive",
            task=task,
            seed=50_000 + length_index,
        )
        tensors = manual_layer0_forward(model, tokens)
        scores = tensors["scores"][0].cpu()
        attention = tensors["attention"][0].cpu()
        target_scores = scores[:, target_index]
        non_target_mask = torch.ones(length, dtype=torch.bool)
        non_target_mask[target_index] = False
        non_target_scores = scores[:, non_target_mask]
        denominator = scores.exp().sum(dim=-1)
        target_exp = target_scores.exp()
        non_target_exp_sum = non_target_scores.exp().sum(dim=-1)
        target_mass = attention[:, target_index]
        query_entropy = entropy(attention)
        max_query = int(target_mass.argmax().item())
        last_query = length - 1

        for query_name, query_index in (
            ("mean_over_queries", None),
            ("target_query", target_index),
            ("last_query", last_query),
            ("max_target_attention_query", max_query),
        ):
            if query_index is None:
                rows.append(
                    {
                        "length": length,
                        "query": query_name,
                        "target_index": target_index,
                        "logit": float(tensors["logit"].cpu().item()),
                        "target_score_mean": float(target_scores.mean().item()),
                        "target_score_max": float(target_scores.max().item()),
                        "non_target_score_mean": float(non_target_scores.mean().item()),
                        "non_target_score_max": float(non_target_scores.max().item()),
                        "target_minus_mean_non_target": float(
                            (target_scores - non_target_scores.mean(dim=-1)).mean().item()
                        ),
                        "target_minus_max_non_target": float(
                            (target_scores - non_target_scores.max(dim=-1).values).mean().item()
                        ),
                        "target_exp_mean": float(target_exp.mean().item()),
                        "non_target_exp_sum_mean": float(non_target_exp_sum.mean().item()),
                        "softmax_denominator_mean": float(denominator.mean().item()),
                        "target_attention_mean": float(target_mass.mean().item()),
                        "target_attention_max": float(target_mass.max().item()),
                        "attention_entropy_mean": float(query_entropy.mean().item()),
                        "attention_entropy_max": float(query_entropy.max().item()),
                    }
                )
                continue

            query_scores = scores[query_index]
            target_score = query_scores[target_index]
            non_target_query_scores = query_scores[non_target_mask]
            rows.append(
                {
                    "length": length,
                    "query": query_name,
                    "target_index": target_index,
                    "query_index": query_index,
                    "logit": float(tensors["logit"].cpu().item()),
                    "target_score": float(target_score.item()),
                    "non_target_score_mean": float(non_target_query_scores.mean().item()),
                    "non_target_score_max": float(non_target_query_scores.max().item()),
                    "target_minus_mean_non_target": float(
                        (target_score - non_target_query_scores.mean()).item()
                    ),
                    "target_minus_max_non_target": float(
                        (target_score - non_target_query_scores.max()).item()
                    ),
                    "target_exp": float(target_score.exp().item()),
                    "non_target_exp_sum": float(non_target_query_scores.exp().sum().item()),
                    "softmax_denominator": float(query_scores.exp().sum().item()),
                    "target_attention": float(attention[query_index, target_index].item()),
                    "attention_entropy": float(query_entropy[query_index].item()),
                }
            )
    return rows


def hidden_evidence_rows(
    model: TransformerTokenPresenceClassifier,
    *,
    task: TaskConfig,
    lengths: list[int],
) -> list[dict[str, object]]:
    """Summarize hidden-state norms and classifier-aligned evidence by length."""

    rows: list[dict[str, object]] = []
    classifier_weight = model.classifier.weight.detach().squeeze(0).cpu()
    tensors_to_inspect = [
        "embedding",
        "attention_pre_out",
        "attention_out",
        "after_attention",
        "feedforward",
        "encoded",
    ]

    for length_index, length in enumerate(lengths):
        tokens, target_index = make_controlled_sequence(
            length=length,
            label_type="positive",
            task=task,
            seed=50_000 + length_index,
        )
        tensors = manual_layer0_forward(model, tokens)
        non_target_mask = torch.ones(length, dtype=torch.bool)
        non_target_mask[target_index] = False

        for tensor_name in tensors_to_inspect:
            values = tensors[tensor_name].detach().cpu()
            target_vector = values[target_index]
            non_target_values = values[non_target_mask]
            evidence = values @ classifier_weight
            rows.append(
                {
                    "length": length,
                    "tensor": tensor_name,
                    "target_index": target_index,
                    "logit": float(tensors["logit"].cpu().item()),
                    "target_norm": float(torch.linalg.vector_norm(target_vector).item()),
                    "non_target_norm_mean": float(
                        torch.linalg.vector_norm(non_target_values, dim=1).mean().item()
                    ),
                    "non_target_norm_max": float(
                        torch.linalg.vector_norm(non_target_values, dim=1).max().item()
                    ),
                    "target_classifier_evidence": float(evidence[target_index].item()),
                    "non_target_classifier_evidence_mean": float(
                        evidence[non_target_mask].mean().item()
                    ),
                    "non_target_classifier_evidence_max": float(
                        evidence[non_target_mask].max().item()
                    ),
                }
            )
    return rows


def maxpool_and_logit_rows(
    model: TransformerTokenPresenceClassifier,
    *,
    task: TaskConfig,
    lengths: list[int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Summarize max-pool source attribution and classifier logit decomposition."""

    maxpool_rows: list[dict[str, object]] = []
    logit_rows: list[dict[str, object]] = []
    classifier_weight = model.classifier.weight.detach().squeeze(0).cpu()
    classifier_bias = float(model.classifier.bias.detach().squeeze().cpu().item())

    for length_index, length in enumerate(lengths):
        tokens, target_index = make_controlled_sequence(
            length=length,
            label_type="positive",
            task=task,
            seed=50_000 + length_index,
        )
        tensors = manual_layer0_forward(model, tokens)
        pooled = tensors["pooled"].detach().cpu()
        argmax_positions = tensors["argmax_positions"].detach().cpu()
        contributions = classifier_weight * pooled
        target_dim_mask = argmax_positions.eq(target_index)
        non_target_dim_mask = ~target_dim_mask
        positive_contributions = contributions.clamp_min(0)
        negative_contributions = contributions.clamp_max(0)

        maxpool_rows.append(
            {
                "length": length,
                "target_index": target_index,
                "logit": float(tensors["logit"].cpu().item()),
                "target_sourced_dim_count": int(target_dim_mask.sum().item()),
                "non_target_sourced_dim_count": int(non_target_dim_mask.sum().item()),
                "target_sourced_dim_fraction": float(target_dim_mask.float().mean().item()),
                "target_sourced_contribution_sum": float(contributions[target_dim_mask].sum().item()),
                "non_target_sourced_contribution_sum": float(
                    contributions[non_target_dim_mask].sum().item()
                ),
                "target_sourced_positive_contribution_sum": float(
                    positive_contributions[target_dim_mask].sum().item()
                ),
                "non_target_sourced_positive_contribution_sum": float(
                    positive_contributions[non_target_dim_mask].sum().item()
                ),
                "target_sourced_negative_contribution_sum": float(
                    negative_contributions[target_dim_mask].sum().item()
                ),
                "non_target_sourced_negative_contribution_sum": float(
                    negative_contributions[non_target_dim_mask].sum().item()
                ),
            }
        )

        top_positive_indices = torch.argsort(contributions, descending=True)[:8]
        top_negative_indices = torch.argsort(contributions)[:8]
        logit_rows.append(
            {
                "length": length,
                "target_index": target_index,
                "logit": float(tensors["logit"].cpu().item()),
                "classifier_bias": classifier_bias,
                "contribution_sum": float(contributions.sum().item()),
                "positive_contribution_sum": float(positive_contributions.sum().item()),
                "negative_contribution_sum": float(negative_contributions.sum().item()),
                "top_positive_dims": json.dumps([int(index) for index in top_positive_indices]),
                "top_positive_contributions": json.dumps(
                    [float(contributions[index].item()) for index in top_positive_indices]
                ),
                "top_positive_sources": json.dumps(
                    [int(argmax_positions[index].item()) for index in top_positive_indices]
                ),
                "top_negative_dims": json.dumps([int(index) for index in top_negative_indices]),
                "top_negative_contributions": json.dumps(
                    [float(contributions[index].item()) for index in top_negative_indices]
                ),
                "top_negative_sources": json.dumps(
                    [int(argmax_positions[index].item()) for index in top_negative_indices]
                ),
            }
        )

    return maxpool_rows, logit_rows
