"""Model definitions for the infinite-length generalization experiments."""

from __future__ import annotations

import torch
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


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
