from __future__ import annotations

from torch import Tensor
from torch.distributions import Distribution, Independent, Normal

from .base import TensorDistribution


class TensorNormal(TensorDistribution):
    loc: Tensor
    scale: Tensor
    reinterpreted_batch_ndims: int = 1

    def dist(self) -> Distribution:
        return Independent(
            Normal(
                loc=self.loc,
                scale=self.scale,
            ),
            self.reinterpreted_batch_ndims,
        )
