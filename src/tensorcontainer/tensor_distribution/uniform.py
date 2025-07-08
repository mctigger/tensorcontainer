from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Distribution, Independent, Uniform

from .base import TensorDistribution


class TensorUniform(TensorDistribution):
    low: Tensor
    high: Tensor
    reinterpreted_batch_ndims: int = 1

    def __post_init__(self):
        super().__post_init__()

        # Validate that low < high
        if torch.any(self.low >= self.high):
            raise ValueError("low must be strictly less than high")

        # Validate that parameters have compatible shapes
        try:
            torch.broadcast_tensors(self.low, self.high)
        except RuntimeError as e:
            raise ValueError(f"low and high must have compatible shapes: {e}")

    def dist(self) -> Distribution:
        return Independent(
            Uniform(
                low=self.low,
                high=self.high,
            ),
            self.reinterpreted_batch_ndims,
        )
