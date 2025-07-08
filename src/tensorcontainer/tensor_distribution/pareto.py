from __future__ import annotations

from dataclasses import field

from torch import Tensor
from torch.distributions import Pareto as TorchPareto
from torch.distributions import Independent

from .base import TensorDistribution


class Pareto(TensorDistribution):
    """
    A Pareto distribution.

    This distribution is parameterized by `scale` and `alpha`.

    Source: https://pytorch.org/docs/stable/distributions.html#pareto
    """

    scale: Tensor = field()
    alpha: Tensor = field()
    reinterpreted_batch_ndims: int = 0

    def dist(self) -> Independent:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return Independent(
            TorchPareto(
                scale=self.scale,
                alpha=self.alpha,
            ),
            self.reinterpreted_batch_ndims,
        )
