from __future__ import annotations

from dataclasses import field

from torch import Tensor
from torch.distributions import InverseGamma as TorchInverseGamma
from torch.distributions import Independent

from .base import TensorDistribution


class InverseGamma(TensorDistribution):
    """
    An Inverse Gamma distribution.

    This distribution is parameterized by `concentration` and `rate`.

    Source: https://pytorch.org/docs/stable/distributions.html#inversegamma
    """

    concentration: Tensor = field()
    rate: Tensor = field()
    reinterpreted_batch_ndims: int = 0

    def dist(self) -> Independent:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return Independent(
            TorchInverseGamma(
                concentration=self.concentration,
                rate=self.rate,
            ),
            self.reinterpreted_batch_ndims,
        )
