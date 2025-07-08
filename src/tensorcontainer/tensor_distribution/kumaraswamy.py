from __future__ import annotations

from dataclasses import field

from torch import Tensor
from torch.distributions import Kumaraswamy as TorchKumaraswamy
from torch.distributions import Independent

from .base import TensorDistribution


class Kumaraswamy(TensorDistribution):
    """
    A Kumaraswamy distribution.

    This distribution is parameterized by `concentration1` and `concentration0`.

    Source: https://pytorch.org/docs/stable/distributions.html#kumaraswamy
    """

    concentration1: Tensor = field()
    concentration0: Tensor = field()
    reinterpreted_batch_ndims: int = 0

    def dist(self) -> Independent:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return Independent(
            TorchKumaraswamy(
                concentration1=self.concentration1,
                concentration0=self.concentration0,
            ),
            self.reinterpreted_batch_ndims,
        )
