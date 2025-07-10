from __future__ import annotations

from torch import Tensor
from torch.distributions import Distribution, Independent, Uniform

from .base import TensorDistribution


class TensorUniform(TensorDistribution):
    low: Tensor
    high: Tensor
    reinterpreted_batch_ndims: int = 1

    def __post_init__(self):
        super().__post_init__()

        # Validate that low < high (PyTorch doesn't validate this with validate_args=False)
        if (self.low >= self.high).any():
            raise ValueError("low must be strictly less than high")

    def dist(self) -> Distribution:
        return Independent(
            Uniform(
                low=self.low,
                high=self.high,
                validate_args=False,
            ),
            self.reinterpreted_batch_ndims,
        )
