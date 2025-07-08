from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Distribution, Independent, Laplace

from .base import TensorDistribution


class TensorLaplace(TensorDistribution):
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
            Laplace(
                loc=self.loc,
                scale=self.scale,
            ),
            self.reinterpreted_batch_ndims,
        )

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the Laplace distribution."""
        return 2 * self.scale**2
