from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import LKJCholesky as TorchLKJCholesky
from torch.distributions.distribution import Distribution

from tensorcontainer.tensor_distribution.base import TensorDistribution


class LKJCholesky(TensorDistribution):
    """
    Creates a distribution of Cholesky factors of correlation matrices.

    The distribution is defined over the space of `d x d` lower-triangular
    matrices `L` with positive diagonal entries, such that `L @ L.T` is a
    correlation matrix.

    Args:
        dimension (int): The dimension of the correlation matrix.
        concentration (Tensor): The concentration parameter of the distribution.
            Must be positive.
    """

    dimension: int
    concentration: Tensor

    def __post_init__(self):
        super().__post_init__()
        if torch.any(self.concentration <= 0):
            raise ValueError("concentration must be positive")

    def dist(self) -> Distribution:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return TorchLKJCholesky(
            dim=self.dimension,
            concentration=self.concentration,
        )
