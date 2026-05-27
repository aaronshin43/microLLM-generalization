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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return one binary logit per sequence."""

        embedded = self.embedding(tokens)
        encoded = self.encoder(embedded)
        pooled = encoded.max(dim=1).values
        return self.classifier(pooled).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
