from __future__ import annotations

from dataclasses import field
from typing import Optional

import torch
from torch import Tensor
from torch.distributions import Binomial, Distribution, Independent

from .base import TensorDistribution


class TensorBinomial(TensorDistribution):
    total_count: Tensor
    probs: Optional[Tensor] = field(default=None)
    logits: Optional[Tensor] = field(default=None)
    reinterpreted_batch_ndims: int = 1

    def __post_init__(self):
        super().__post_init__()

        # Validate that exactly one of probs or logits is provided
        if (self.probs is None) == (self.logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )

        # Validate total_count is a non-negative integer
        if torch.any(self.total_count < 0) or torch.any(
            self.total_count != self.total_count.int()
        ):
            raise ValueError("total_count must be a non-negative integer")

        # Convert total_count to float for torch distribution
        self.total_count = self.total_count.float()

        # Validate probs if provided
        if self.probs is not None:
            if torch.any(self.probs < 0) or torch.any(self.probs > 1):
                raise ValueError("probs must be in the range [0, 1]")

    def dist(self) -> Distribution:
        if self.probs is not None:
            return Independent(
                Binomial(
                    total_count=self.total_count,
                    probs=self.probs,
                ),
                self.reinterpreted_batch_ndims,
            )
        else:
            return Independent(
                Binomial(
                    total_count=self.total_count,
                    logits=self.logits,
                ),
                self.reinterpreted_batch_ndims,
            )

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the Binomial distribution."""
        if self.probs is not None:
            p = self.probs
        else:
            assert self.logits is not None
            p = torch.sigmoid(self.logits)
        return self.total_count * p * (1 - p)
