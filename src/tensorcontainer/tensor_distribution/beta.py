from __future__ import annotations

from torch import Tensor
from torch.distributions import Beta, Distribution, Independent

from .base import TensorDistribution


class TensorBeta(TensorDistribution):
    concentration1: Tensor
    concentration0: Tensor
    reinterpreted_batch_ndims: int = 1

    def dist(self) -> Distribution:
        return Independent(
            Beta(
                concentration1=self.concentration1,
                concentration0=self.concentration0,
            ),
            self.reinterpreted_batch_ndims,
        )

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the Beta distribution."""
        a = self.concentration1
        b = self.concentration0
        return (a * b) / ((a + b) ** 2 * (a + b + 1))
