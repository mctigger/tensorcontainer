from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.distributions import (
    LowRankMultivariateNormal as TorchLowRankMultivariateNormal,
)

from tensorcontainer.tensor_distribution.base import TensorDistribution


class TensorLowRankMultivariateNormal(TensorDistribution):
    """
    Creates a multivariate normal distribution with a low-rank covariance matrix.

    Args:
        loc (Tensor): Mean of the distribution.
        cov_factor (Tensor): Factor part of low-rank form of covariance matrix.
        cov_diag (Tensor): Diagonal part of low-rank form of covariance matrix.
    """

    _loc: Tensor
    _cov_factor: Tensor
    _cov_diag: Tensor

    def __init__(
        self,
        loc: Tensor,
        cov_factor: Tensor,
        cov_diag: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self._loc = loc
        self._cov_factor = cov_factor
        self._cov_diag = cov_diag

        super().__init__(
            shape=loc.shape[:-1], device=loc.device, validate_args=validate_args
        )

    def dist(self) -> TorchLowRankMultivariateNormal:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return TorchLowRankMultivariateNormal(
            loc=self._loc,
            cov_factor=self._cov_factor,
            cov_diag=self._cov_diag,
            validate_args=self._validate_args,
        )

    @property
    def covariance_matrix(self) -> Tensor:
        """Covariance matrix of the distribution."""
        return self.dist().covariance_matrix

    @property
    def precision_matrix(self) -> Tensor:
        """Precision matrix of the distribution."""
        return self.dist().precision_matrix

    @property
    def scale_tril(self) -> Tensor:
        """Lower-triangular factor of the covariance matrix."""
        return self.dist().scale_tril

    @classmethod
    def _unflatten_distribution(
        cls, attributes: dict[str, Any]
    ) -> TensorLowRankMultivariateNormal:
        return TensorLowRankMultivariateNormal(
            loc=attributes["_loc"],
            cov_factor=attributes["_cov_factor"],
            cov_diag=attributes["_cov_diag"],
            validate_args=attributes.get("_validate_args"),
        )
