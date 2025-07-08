from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Chi2, Distribution, Independent

from .base import TensorDistribution


class TensorChi2(TensorDistribution):
    df: Tensor
    reinterpreted_batch_ndims: int = 1

    def __post_init__(self):
        super().__post_init__()

        if torch.any(self.df <= 0):
            raise ValueError("df must be positive")

    def dist(self) -> Distribution:
        return Independent(
            Chi2(df=self.df),
            self.reinterpreted_batch_ndims,
        )

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the Chi2 distribution."""
        return 2 * self.df
