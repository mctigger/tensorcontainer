from __future__ import annotations

from torch import Tensor
from torch.distributions import Distribution, Independent, Laplace

from .base import TensorDistribution


class TensorLaplace(TensorDistribution):
    loc: Tensor
    scale: Tensor
    reinterpreted_batch_ndims: int = 1

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
