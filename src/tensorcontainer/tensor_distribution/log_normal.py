from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Distribution, Independent, LogNormal

from .base import TensorDistribution


class TensorLogNormal(TensorDistribution):
    loc: Tensor
    scale: Tensor
    reinterpreted_batch_ndims: int = 1

    def __post_init__(self):
        super().__post_init__()

        if torch.any(self.scale <= 0):
            raise ValueError("scale must be positive")

        try:
            torch.broadcast_tensors(self.loc, self.scale)
        except RuntimeError as e:
            raise ValueError(f"loc and scale must have compatible shapes: {e}")

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
