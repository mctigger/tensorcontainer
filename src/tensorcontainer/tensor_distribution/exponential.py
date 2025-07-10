from __future__ import annotations

from torch import Tensor
from torch.distributions import Distribution, Exponential, Independent

from .base import TensorDistribution


class TensorExponential(TensorDistribution):
    rate: Tensor
    reinterpreted_batch_ndims: int = 1

    def dist(self) -> Distribution:
        return Independent(
            Exponential(
                rate=self.rate,
            ),
            self.reinterpreted_batch_ndims,
        )

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the Exponential distribution."""
        return 1 / (self.rate**2)
