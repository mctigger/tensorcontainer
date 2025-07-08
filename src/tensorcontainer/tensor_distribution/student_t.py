from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Distribution, Independent, StudentT

from .base import TensorDistribution


class TensorStudentT(TensorDistribution):
    df: Tensor
    loc: Tensor
    scale: Tensor
    reinterpreted_batch_ndims: int = 1

    def __post_init__(self):
        super().__post_init__()

        if torch.any(self.df <= 0):
            raise ValueError("df must be positive")
        if torch.any(self.scale <= 0):
            raise ValueError("scale must be positive")

        try:
            torch.broadcast_tensors(self.df, self.loc, self.scale)
        except RuntimeError as e:
            raise ValueError(f"df, loc, and scale must have compatible shapes: {e}")

    def dist(self) -> Distribution:
        return Independent(
            StudentT(df=self.df, loc=self.loc, scale=self.scale),
            self.reinterpreted_batch_ndims,
        )

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the StudentT distribution."""
        var = torch.full_like(self.df, float("inf"))
        var[self.df > 2] = (self.scale**2 * (self.df / (self.df - 2)))[self.df > 2]
        var[self.df <= 1] = float("nan")
        return var
