from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Dirichlet, Distribution

from .base import TensorDistribution


class TensorDirichlet(TensorDistribution):
    concentration: Tensor
    reinterpreted_batch_ndims: int = 0

    def __post_init__(self):
        super().__post_init__()

        # Validate that concentration is positive
        if torch.any(self.concentration <= 0):
            raise ValueError("concentration must be positive")

    def dist(self) -> Distribution:
        return Dirichlet(
            concentration=self.concentration,
        )

    @property
    def batch_shape(self) -> torch.Size:
        """Returns the batch shape of the distribution."""
        return self.concentration.shape[:-1]

    @property
    def event_shape(self) -> torch.Size:
        """Returns the event shape of the distribution."""
        return self.concentration.shape[-1:]

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the Dirichlet distribution."""
        # For Dirichlet, variance is a matrix, but we return the diagonal (marginal variances)
        alpha = self.concentration
        alpha_sum = alpha.sum(-1, keepdim=True)
        return (alpha * (alpha_sum - alpha)) / (alpha_sum**2 * (alpha_sum + 1))
