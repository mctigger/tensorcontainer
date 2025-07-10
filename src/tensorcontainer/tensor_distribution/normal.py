from __future__ import annotations

from torch import Tensor
from torch.distributions import Independent, Normal

from .base import TensorDistribution


class TensorNormal(TensorDistribution):
    loc: Tensor
    scale: Tensor
    reinterpreted_batch_ndims: int = 1

    def dist(self) -> Independent:
        return Independent(
            Normal(loc=self.loc, scale=self.scale, validate_args=False),
            self.reinterpreted_batch_ndims,
        )

    def cdf(self, value: Tensor) -> Tensor:
        return Normal(loc=self.loc, scale=self.scale, validate_args=False).cdf(value)

    def icdf(self, value: Tensor) -> Tensor:
        return Normal(loc=self.loc, scale=self.scale, validate_args=False).icdf(value)

    @property
    def variance(self) -> Tensor:
        return self.dist().variance
