from __future__ import annotations

from dataclasses import field

from torch import Tensor
from torch.distributions import Gumbel as TorchGumbel
from torch.distributions import Independent

from .base import TensorDistribution


class Gumbel(TensorDistribution):
    """
    A Gumbel distribution.

    This distribution is parameterized by `loc` and `scale`.

    Source: https://pytorch.org/docs/stable/distributions.html#gumbel
    """

    loc: Tensor = field()
    scale: Tensor = field()
    reinterpreted_batch_ndims: int = 0

    def dist(self) -> Independent:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return Independent(
            TorchGumbel(
                loc=self.loc,
                scale=self.scale,
            ),
            self.reinterpreted_batch_ndims,
        )
