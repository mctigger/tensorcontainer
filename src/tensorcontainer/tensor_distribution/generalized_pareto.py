from __future__ import annotations

from dataclasses import field

from torch import Tensor
from torch.distributions import GeneralizedPareto as TorchGeneralizedPareto
from torch.distributions import Independent

from .base import TensorDistribution


class GeneralizedPareto(TensorDistribution):
    """
    A Generalized Pareto distribution.

    This distribution is parameterized by `scale` and `concentration`.

    Source: https://pytorch.org/docs/stable/distributions.html#generalizedpareto
    """

    scale: Tensor = field()
    concentration: Tensor = field()
    reinterpreted_batch_ndims: int = 0

    def dist(self) -> Independent:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return Independent(
            TorchGeneralizedPareto(
                scale=self.scale,
                concentration=self.concentration,
            ),
            self.reinterpreted_batch_ndims,
        )
