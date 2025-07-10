from __future__ import annotations

from torch import Tensor
from torch.distributions import Distribution, Gamma, Independent

from .base import TensorDistribution


class TensorGamma(TensorDistribution):
    concentration: Tensor
    rate: Tensor
    reinterpreted_batch_ndims: int = 1

    def dist(self) -> Distribution:
        return Independent(
            Gamma(
                concentration=self.concentration,
                rate=self.rate,
            ),
            self.reinterpreted_batch_ndims,
        )

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the Gamma distribution."""
        return self.concentration / (self.rate**2)
