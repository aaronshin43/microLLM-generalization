"""Model definitions for the infinite-length generalization experiments."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class MaxPoolTokenPresenceClassifier(nn.Module):
    """Permutation-invariant baseline for existential token detection.

    The model applies the same detector to every token representation, then uses max
    pooling across the sequence. There are no length-specific parameters.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return one binary logit per sequence."""

        embedded = self.embedding(tokens)
        token_features = self.token_mlp(embedded)
        pooled = token_features.max(dim=1).values
        return self.classifier(pooled).squeeze(-1)


class TransformerTokenPresenceClassifier(nn.Module):
    """Minimal no-position transformer for existential token detection.

    The model intentionally omits positional encodings and uses max pooling so that
    it has no learned parameters tied to the training sequence length.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                AttentionExportEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Linear(d_model, 1)

    def encode(
        self,
        tokens: torch.Tensor,
        *,
        return_attention: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode tokens and optionally return per-layer attention weights."""

        hidden = self.embedding(tokens)
        attention_weights: list[torch.Tensor] = []
        for layer in self.layers:
            hidden, weights = layer(hidden, return_attention=return_attention)
            if weights is not None:
                attention_weights.append(weights)
        return hidden, attention_weights

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return one binary logit per sequence."""

        encoded, _ = self.encode(tokens, return_attention=False)
        pooled = encoded.max(dim=1).values
        return self.classifier(pooled).squeeze(-1)

    def forward_with_attention(
        self,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Return logits, pooled activations, and attention weights for analysis.

        Each attention tensor has shape `(batch, heads, query_length, key_length)`.
        """

        encoded, attention_weights = self.encode(tokens, return_attention=True)
        pooled = encoded.max(dim=1).values
        logits = self.classifier(pooled).squeeze(-1)
        return logits, pooled, attention_weights


class AttentionExportEncoderLayer(nn.Module):
    """Small post-norm transformer encoder layer that can expose attention weights."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        return_attention: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run one encoder layer and optionally return self-attention weights."""

        attention_output, attention_weights = self.self_attn(
            hidden,
            hidden,
            hidden,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        hidden = self.norm1(hidden + self.dropout1(attention_output))
        feedforward_output = self.linear2(self.dropout(self.activation(self.linear1(hidden))))
        hidden = self.norm2(hidden + self.dropout2(feedforward_output))
        return hidden, attention_weights


class LengthAwareTransformerTokenPresenceClassifier(nn.Module):
    """Transformer classifier with learned length-aware attention corrections."""

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        attention_variant: str,
        log_scale_init: float,
        target_detector: str,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                LengthAwareAttentionEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    attention_variant=attention_variant,
                    log_scale_init=log_scale_init,
                    target_detector=target_detector,
                )
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Linear(d_model, 1)

    def encode(
        self,
        tokens: torch.Tensor,
        *,
        return_attention: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode tokens and optionally return per-layer attention weights."""

        hidden = self.embedding(tokens)
        attention_weights: list[torch.Tensor] = []
        for layer in self.layers:
            hidden, weights = layer(hidden, return_attention=return_attention)
            if weights is not None:
                attention_weights.append(weights)
        return hidden, attention_weights

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return one binary logit per sequence."""

        encoded, _ = self.encode(tokens, return_attention=False)
        pooled = encoded.max(dim=1).values
        return self.classifier(pooled).squeeze(-1)

    def forward_with_attention(
        self,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Return logits, pooled activations, and attention weights for analysis."""

        encoded, attention_weights = self.encode(tokens, return_attention=True)
        pooled = encoded.max(dim=1).values
        logits = self.classifier(pooled).squeeze(-1)
        return logits, pooled, attention_weights

    def length_scale_rows(self, lengths: tuple[int, ...]) -> list[dict[str, float | int | str]]:
        """Return learned length-scale values for audit logging."""

        rows: list[dict[str, float | int | str]] = []
        for layer_index, layer in enumerate(self.layers):
            rows.extend(layer.length_scale_rows(lengths=lengths, layer_index=layer_index))
        return rows


class LengthAwareAttentionEncoderLayer(nn.Module):
    """Post-norm transformer layer with custom length-aware self-attention."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        attention_variant: str,
        log_scale_init: float,
        target_detector: str,
    ) -> None:
        super().__init__()
        self.self_attn = LengthAwareSelfAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            attention_variant=attention_variant,
            log_scale_init=log_scale_init,
            target_detector=target_detector,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        return_attention: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run one encoder layer and optionally return self-attention weights."""

        attention_output, attention_weights = self.self_attn(
            hidden,
            return_attention=return_attention,
        )
        hidden = self.norm1(hidden + self.dropout1(attention_output))
        feedforward_output = self.linear2(self.dropout(self.activation(self.linear1(hidden))))
        hidden = self.norm2(hidden + self.dropout2(feedforward_output))
        return hidden, attention_weights

    def length_scale_rows(
        self,
        *,
        lengths: tuple[int, ...],
        layer_index: int,
    ) -> list[dict[str, float | int | str]]:
        """Return learned attention length-scale values for this layer."""

        return self.self_attn.length_scale_rows(lengths=lengths, layer_index=layer_index)


class LengthAwareSelfAttention(nn.Module):
    """Self-attention with learned log-length score corrections.

    The module keeps the same projection structure as PyTorch MultiheadAttention
    but exposes the logits so Stage 2B can apply length-aware corrections.
    """

    VALID_VARIANTS = {"global_log_temperature", "target_key_log_bias"}
    VALID_TARGET_DETECTORS = {"linear"}

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        attention_variant: str,
        log_scale_init: float,
        target_detector: str,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if attention_variant not in self.VALID_VARIANTS:
            raise ValueError(f"unknown attention_variant: {attention_variant}")
        if target_detector not in self.VALID_TARGET_DETECTORS:
            raise ValueError(f"unknown target_detector: {target_detector}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attention_variant = attention_variant
        self.target_detector_name = target_detector
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.log_scale = nn.Parameter(torch.tensor(float(log_scale_init)))

        if attention_variant == "target_key_log_bias":
            self.target_detector = nn.Linear(embed_dim, 1)
        else:
            self.target_detector = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize projections using PyTorch MultiheadAttention-like defaults."""

        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        if self.target_detector is not None:
            nn.init.xavier_uniform_(self.target_detector.weight)
            nn.init.constant_(self.target_detector.bias, 0.0)

    def length_factor(self, sequence_length: int, *, device: torch.device) -> torch.Tensor:
        """Return `log1p(length)` as a tensor on the attention device."""

        return torch.log1p(torch.tensor(float(sequence_length), device=device))

    def positive_scale(self) -> torch.Tensor:
        """Return the positive learned scalar used by the length correction."""

        return F.softplus(self.log_scale)

    def correction_value(self, sequence_length: int, *, device: torch.device) -> torch.Tensor:
        """Return the learned positive scale times `log1p(sequence_length)`."""

        return self.positive_scale() * self.length_factor(sequence_length, device=device)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        return_attention: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply length-aware self-attention.

        Returned attention weights have shape `(batch, heads, query, key)`.
        """

        batch_size, sequence_length, _ = hidden.shape
        q_full, k_full, v_full = F.linear(
            hidden,
            self.in_proj_weight,
            self.in_proj_bias,
        ).chunk(3, dim=-1)

        q = self._split_heads(q_full)
        k = self._split_heads(k_full)
        v = self._split_heads(v_full)
        base_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        corrected_scores = self.apply_length_correction(
            base_scores=base_scores,
            k_full=k_full,
            sequence_length=sequence_length,
        )
        attention_weights = torch.softmax(corrected_scores, dim=-1)
        dropped_attention = self.attention_dropout(attention_weights)
        context = torch.matmul(dropped_attention, v)
        merged_context = context.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            self.embed_dim,
        )
        output = self.out_proj(merged_context)
        if return_attention:
            return output, attention_weights
        return output, None

    def apply_length_correction(
        self,
        *,
        base_scores: torch.Tensor,
        k_full: torch.Tensor,
        sequence_length: int,
    ) -> torch.Tensor:
        """Apply the configured length-aware correction to attention logits."""

        correction = self.correction_value(sequence_length, device=base_scores.device)
        if self.attention_variant == "global_log_temperature":
            return (1.0 + correction) * base_scores

        if self.target_detector is None:
            raise RuntimeError("target_key_log_bias requires a target detector")
        target_like = self.target_detector(k_full).squeeze(-1)
        return base_scores + correction * target_like[:, None, None, :]

    def _split_heads(self, values: torch.Tensor) -> torch.Tensor:
        """Convert `(batch, length, embed)` into `(batch, heads, length, head_dim)`."""

        batch_size, sequence_length, _ = values.shape
        return values.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def length_scale_rows(
        self,
        *,
        lengths: tuple[int, ...],
        layer_index: int,
    ) -> list[dict[str, float | int | str]]:
        """Return learned length-scale values for CSV-friendly audit logging."""

        rows: list[dict[str, float | int | str]] = []
        device = self.log_scale.device
        learned_scale = float(self.positive_scale().detach().cpu().item())
        for length in lengths:
            correction = float(
                self.correction_value(length, device=device).detach().cpu().item()
            )
            row: dict[str, float | int | str] = {
                "layer": layer_index,
                "attention_variant": self.attention_variant,
                "length": length,
                "learned_positive_scale": learned_scale,
                "log1p_length": float(math.log1p(length)),
                "length_correction": correction,
            }
            if self.attention_variant == "global_log_temperature":
                row["alpha_length_scale"] = 1.0 + correction
            else:
                row["beta_length_scale"] = correction
            rows.append(row)
        return rows


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_rows(model: nn.Module) -> list[dict[str, object]]:
    """Return compact descriptions for every trainable parameter."""

    rows: list[dict[str, object]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        rows.append(
            {
                "name": name,
                "shape": tuple(parameter.shape),
                "dtype": str(parameter.dtype),
                "device": str(parameter.device),
                "count": parameter.numel(),
            }
        )
    return rows


def format_trainable_parameters(model: nn.Module) -> str:
    """Format trainable parameter descriptions as a readable table."""

    rows = trainable_parameter_rows(model)
    if not rows:
        return "No trainable parameters."

    name_width = max(len(str(row["name"])) for row in rows)
    shape_width = max(len(str(row["shape"])) for row in rows)
    dtype_width = max(len(str(row["dtype"])) for row in rows)
    device_width = max(len(str(row["device"])) for row in rows)

    lines = [
        "Trainable parameters:",
        (
            f"{'name':{name_width}}  {'shape':{shape_width}}  "
            f"{'dtype':{dtype_width}}  {'device':{device_width}}  count"
        ),
        (
            f"{'-' * name_width}  {'-' * shape_width}  "
            f"{'-' * dtype_width}  {'-' * device_width}  -----"
        ),
    ]
    total = 0
    for row in rows:
        count = int(row["count"])
        total += count
        lines.append(
            f"{str(row['name']):{name_width}}  {str(row['shape']):{shape_width}}  "
            f"{str(row['dtype']):{dtype_width}}  {str(row['device']):{device_width}}  {count}"
        )
    lines.append(f"Total trainable parameters: {total}")
    return "\n".join(lines)
