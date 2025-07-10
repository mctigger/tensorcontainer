from __future__ import annotations

from dataclasses import field
from typing import Optional

import torch
from torch import Tensor
from torch.distributions import Distribution, Geometric, Independent

from .base import TensorDistribution


class TensorGeometric(TensorDistribution):
    probs: Optional[Tensor] = field(default=None)
    logits: Optional[Tensor] = field(default=None)
    reinterpreted_batch_ndims: int = 1

    def dist(self) -> Distribution:
        if self.probs is not None:
            return Independent(
                Geometric(
                    probs=self.probs,
                ),
                self.reinterpreted_batch_ndims,
            )
        else:
            return Independent(
                Geometric(
                    logits=self.logits,
                ),
                self.reinterpreted_batch_ndims,
            )

    @property
    def mean(self) -> Tensor:
        """Returns the mean of the Geometric distribution."""
        if self.probs is not None:
            p = self.probs
        else:
            assert self.logits is not None
            p = torch.sigmoid(self.logits)
        return 1 / p

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the Geometric distribution."""
        if self.probs is not None:
            p = self.probs
        else:
            assert self.logits is not None
            p = torch.sigmoid(self.logits)
        return (1 - p) / (p**2)
