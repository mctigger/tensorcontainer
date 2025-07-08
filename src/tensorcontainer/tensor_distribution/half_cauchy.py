from __future__ import annotations

from dataclasses import field

from torch import Tensor
from torch.distributions import HalfCauchy as TorchHalfCauchy
from torch.distributions import Independent

from .base import TensorDistribution


class HalfCauchy(TensorDistribution):
    """
    A Half-Cauchy distribution.

    This distribution is parameterized by `scale`.

    Source: https://pytorch.org/docs/stable/distributions.html#halfcauchy
    """

    scale: Tensor = field()
    reinterpreted_batch_ndims: int = 0

    def dist(self) -> Independent:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return Independent(
            TorchHalfCauchy(
                scale=self.scale,
            ),
            self.reinterpreted_batch_ndims,
        )
