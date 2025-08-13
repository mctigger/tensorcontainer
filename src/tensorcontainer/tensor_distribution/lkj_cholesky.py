from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.distributions import LKJCholesky as TorchLKJCholesky
from torch.distributions.distribution import Distribution

from tensorcontainer.tensor_distribution.base import TensorDistribution
from .utils import broadcast_all


class TensorLKJCholesky(TensorDistribution):
    """
    Creates a distribution of Cholesky factors of correlation matrices.

    The distribution is defined over the space of `d x d` lower-triangular
    matrices `L` with positive diagonal entries, such that `L @ L.T` is a
    correlation matrix.

    Args:
        dim (int): The dimension of the correlation matrix.
        concentration (float or Tensor): The concentration parameter of the distribution.
            Must be positive.
    """

    _dim: int
    _concentration: Tensor

    def __init__(
        self,
        dim: int,
        concentration: float | Tensor = 1.0,
        validate_args: bool | None = None,
    ) -> None:
        self._dim = dim
        (self._concentration,) = broadcast_all(concentration)
        super().__init__(
            shape=self._concentration.shape,
            device=self._concentration.device,
            validate_args=validate_args,
        )

    def dist(self) -> Distribution:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return TorchLKJCholesky(
            dim=self._dim,
            concentration=self._concentration,
            validate_args=self._validate_args,
        )

    @classmethod
    def _unflatten_distribution(cls, attributes: dict[str, Any]) -> TensorLKJCholesky:
        return cls(
            dim=attributes["_dim"],
            concentration=attributes["_concentration"],
            validate_args=attributes.get("_validate_args"),
        )
