from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal

from .base import TensorDistribution


class TensorMultivariateNormal(TensorDistribution):
    loc: Tensor
    covariance_matrix: Tensor
    reinterpreted_batch_ndims: int = 0

    def __post_init__(self):
        super().__post_init__()

        # Validate that loc and covariance_matrix have compatible shapes
        if self.loc.shape[-1] != self.covariance_matrix.shape[-1]:
            raise ValueError(
                f"loc and covariance_matrix must have compatible shapes: "
                f"loc.shape={self.loc.shape}, covariance_matrix.shape={self.covariance_matrix.shape}"
            )

        if self.covariance_matrix.shape[-2] != self.covariance_matrix.shape[-1]:
            raise ValueError("covariance_matrix must be square")

        # Validate that covariance matrix is positive definite
        try:
            torch.linalg.cholesky(self.covariance_matrix)
        except RuntimeError:
            raise ValueError("covariance_matrix must be positive definite")

    def dist(self) -> Distribution:
        return MultivariateNormal(
            loc=self.loc,
            covariance_matrix=self.covariance_matrix,
        )

    @property
    def batch_shape(self) -> torch.Size:
        """Returns the batch shape of the distribution."""
        return self.loc.shape[:-1]

    @property
    def event_shape(self) -> torch.Size:
        """Returns the event shape of the distribution."""
        return self.loc.shape[-1:]

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the MultivariateNormal distribution."""
        return torch.diagonal(self.covariance_matrix, dim1=-2, dim2=-1)
