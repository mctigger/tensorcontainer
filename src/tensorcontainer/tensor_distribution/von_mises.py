from __future__ import annotations

from dataclasses import field

from torch import Tensor
from torch.distributions import VonMises as TorchVonMises
from torch.distributions import Independent

from .base import TensorDistribution


class VonMises(TensorDistribution):
    """
    A Von Mises distribution.

    This distribution is parameterized by `loc` and `concentration`.

    Source: https://pytorch.org/docs/stable/distributions.html#vonmises
    """

    loc: Tensor = field()
    concentration: Tensor = field()
    reinterpreted_batch_ndims: int = 0

    def dist(self) -> Independent:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return Independent(
            TorchVonMises(
                loc=self.loc,
                concentration=self.concentration,
            ),
            self.reinterpreted_batch_ndims,
        )
