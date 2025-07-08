from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Beta, Distribution, Independent

from .base import TensorDistribution


class TensorBeta(TensorDistribution):
    concentration1: Tensor
    concentration0: Tensor
    reinterpreted_batch_ndims: int = 1

    def __post_init__(self):
        super().__post_init__()

        # Validate that concentration parameters are positive
        if torch.any(self.concentration1 <= 0):
            raise ValueError("concentration1 must be positive")
        if torch.any(self.concentration0 <= 0):
            raise ValueError("concentration0 must be positive")

        # Validate that parameters have compatible shapes
        try:
            torch.broadcast_tensors(self.concentration1, self.concentration0)
        except RuntimeError as e:
            raise ValueError(
                f"concentration1 and concentration0 must have compatible shapes: {e}"
            )

    def dist(self) -> Distribution:
        return Independent(
            Beta(
                concentration1=self.concentration1,
                concentration0=self.concentration0,
            ),
            self.reinterpreted_batch_ndims,
        )

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the Beta distribution."""
        a = self.concentration1
        b = self.concentration0
        return (a * b) / ((a + b) ** 2 * (a + b + 1))
