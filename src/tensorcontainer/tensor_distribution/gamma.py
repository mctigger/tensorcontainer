from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Distribution, Gamma, Independent

from .base import TensorDistribution


class TensorGamma(TensorDistribution):
    concentration: Tensor
    rate: Tensor
    reinterpreted_batch_ndims: int = 1

    def __post_init__(self):
        super().__post_init__()

        # Validate that concentration and rate are positive
        if torch.any(self.concentration <= 0):
            raise ValueError("concentration must be positive")
        if torch.any(self.rate <= 0):
            raise ValueError("rate must be positive")

        # Validate that parameters have compatible shapes
        try:
            torch.broadcast_tensors(self.concentration, self.rate)
        except RuntimeError as e:
            raise ValueError(f"concentration and rate must have compatible shapes: {e}")

    def dist(self) -> Distribution:
        return Independent(
            Gamma(
                concentration=self.concentration,
                rate=self.rate,
            ),
            self.reinterpreted_batch_ndims,
        )

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the Gamma distribution."""
        return self.concentration / (self.rate**2)
