from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Distribution, Independent, Poisson

from .base import TensorDistribution


class TensorPoisson(TensorDistribution):
    rate: Tensor
    reinterpreted_batch_ndims: int = 1

    def __post_init__(self):
        super().__post_init__()

        # Validate that rate is positive
        if torch.any(self.rate <= 0):
            raise ValueError("rate must be positive")

    def dist(self) -> Distribution:
        return Independent(
            Poisson(
                rate=self.rate,
            ),
            self.reinterpreted_batch_ndims,
        )

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the Poisson distribution."""
        return self.rate
