from __future__ import annotations

from dataclasses import field

from torch import Tensor
from torch.distributions import Weibull as TorchWeibull
from torch.distributions import Independent

from .base import TensorDistribution


class Weibull(TensorDistribution):
    """
    A Weibull distribution.

    This distribution is parameterized by `scale` and `concentration`.

    Source: https://pytorch.org/docs/stable/distributions.html#weibull
    """

    scale: Tensor = field()
    concentration: Tensor = field()
    reinterpreted_batch_ndims: int = 0

    def dist(self) -> Independent:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return Independent(
            TorchWeibull(
                scale=self.scale,
                concentration=self.concentration,
            ),
            self.reinterpreted_batch_ndims,
        )
