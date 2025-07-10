from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Distribution, Independent, LogNormal

from .base import TensorDistribution


class TensorLogNormal(TensorDistribution):
    loc: Tensor
    scale: Tensor
    reinterpreted_batch_ndims: int = 1

    def dist(self) -> Distribution:
        return Independent(
            LogNormal(
                loc=self.loc,
                scale=self.scale,
            ),
            self.reinterpreted_batch_ndims,
        )

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the LogNormal distribution."""
        return (torch.exp(self.scale**2) - 1) * torch.exp(2 * self.loc + self.scale**2)
