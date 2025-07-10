from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Distribution, HalfNormal, Independent

from .base import TensorDistribution


class TensorHalfNormal(TensorDistribution):
    scale: Tensor
    reinterpreted_batch_ndims: int = 1

    def dist(self) -> Distribution:
        return Independent(
            HalfNormal(scale=self.scale),
            self.reinterpreted_batch_ndims,
        )

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the HalfNormal distribution."""
        return self.scale**2 * (1 - 2 / torch.pi)
