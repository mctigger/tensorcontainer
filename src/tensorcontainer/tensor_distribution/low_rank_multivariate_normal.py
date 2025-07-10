from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import (
    LowRankMultivariateNormal as TorchLowRankMultivariateNormal,
)
from torch.distributions.distribution import Distribution

from tensorcontainer.tensor_distribution.base import TensorDistribution


class LowRankMultivariateNormal(TensorDistribution):
    """
    Creates a multivariate normal distribution with a low-rank covariance matrix.

    Args:
        loc (Tensor): The mean of the distribution.
        cov_factor (Tensor): The covariance factor of the distribution.
        cov_diag (Tensor): The diagonal of the covariance matrix.
    """

    loc: Tensor
    cov_factor: Tensor
    cov_diag: Tensor

    def __post_init__(self):
        # Validate cov_diag before calling super().__post_init__()
        # which will call self.dist() and potentially fail with Cholesky error
        if torch.any(self.cov_diag <= 0):
            raise ValueError("cov_diag must be positive")
        super().__post_init__()

    def dist(self) -> Distribution:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return TorchLowRankMultivariateNormal(
            loc=self.loc,
            cov_factor=self.cov_factor,
            cov_diag=self.cov_diag,
        )
