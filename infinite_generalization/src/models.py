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


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
