from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.distributions import Distribution, Independent, NegativeBinomial

from .base import TensorDistribution


class TensorNegativeBinomial(TensorDistribution):
    total_count: Tensor
    probs: Optional[Tensor] = None
    logits: Optional[Tensor] = None
    reinterpreted_batch_ndims: int = 1

    def __post_init__(self):
        super().__post_init__()

        if torch.any(self.total_count <= 0):
            raise ValueError("total_count must be positive")

        if self.probs is None and self.logits is None:
            raise ValueError("Either probs or logits must be specified")
        if self.probs is not None and self.logits is not None:
            raise ValueError("Only one of probs or logits can be specified")

        param = self.probs if self.probs is not None else self.logits
        try:
            torch.broadcast_tensors(self.total_count, param)
        except RuntimeError as e:
            raise ValueError(
                f"total_count and probs/logits must have compatible shapes: {e}"
            )

    def dist(self) -> Distribution:
        return Independent(
            NegativeBinomial(
                total_count=self.total_count, probs=self.probs, logits=self.logits
            ),
            self.reinterpreted_batch_ndims,
        )

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the NegativeBinomial distribution."""
        return self.dist().variance
