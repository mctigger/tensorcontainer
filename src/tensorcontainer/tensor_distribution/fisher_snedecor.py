from __future__ import annotations

from dataclasses import field

from torch import Tensor
from torch.distributions import FisherSnedecor as TorchFisherSnedecor
from torch.distributions import Independent

from .base import TensorDistribution


class FisherSnedecor(TensorDistribution):
    """
    A Fisher-Snedecor distribution.

    This distribution is parameterized by two degrees of freedom parameters, `df1` and `df2`.

    Source: https://pytorch.org/docs/stable/distributions.html#fishersnedecor
    """

    df1: Tensor = field()
    df2: Tensor = field()
    reinterpreted_batch_ndims: int = 0

    def dist(self) -> Independent:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return Independent(
            TorchFisherSnedecor(
                df1=self.df1,
                df2=self.df2,
            ),
            self.reinterpreted_batch_ndims,
        )
