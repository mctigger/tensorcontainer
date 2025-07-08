from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.distributions import Wishart as TorchWishart
from torch.distributions.distribution import Distribution

from tensorcontainer.tensor_distribution.base import TensorDistribution


class Wishart(TensorDistribution):
    """
    Creates a Wishart distribution parameterized by a symmetric positive definite matrix.

    Args:
        df (Tensor): The degrees of freedom of the distribution.
        covariance_matrix (Optional[Tensor]): The covariance matrix of the distribution.
        precision_matrix (Optional[Tensor]): The precision matrix of the distribution.
        scale_tril (Optional[Tensor]): The lower-triangular Cholesky factor of the scale matrix.
    """

    df: Tensor
    covariance_matrix: Optional[Tensor] = None
    precision_matrix: Optional[Tensor] = None
    scale_tril: Optional[Tensor] = None

    def __post_init__(self):
        super().__post_init__()
        if torch.any(self.df <= 0):
            raise ValueError("df must be positive")

    def dist(self) -> Distribution:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return TorchWishart(
            df=self.df,
            covariance_matrix=self.covariance_matrix,
            precision_matrix=self.precision_matrix,
            scale_tril=self.scale_tril,
        )
