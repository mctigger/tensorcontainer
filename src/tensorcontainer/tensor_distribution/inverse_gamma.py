from __future__ import annotations

from typing import Any, Dict, Optional, Union

from torch import Tensor
from torch.distributions import InverseGamma as TorchInverseGamma
from torch.distributions.utils import broadcast_all

from .base import TensorDistribution


class TensorInverseGamma(TensorDistribution):
    """Tensor-aware Inverse Gamma distribution."""

    _concentration: Tensor
    _rate: Tensor

    def __init__(
        self,
        concentration: Union[float, Tensor],
        rate: Union[float, Tensor],
        validate_args: Optional[bool] = None,
    ):
        self._concentration, self._rate = broadcast_all(concentration, rate)

        shape = self._concentration.shape
        device = self._concentration.device

        super().__init__(shape, device, validate_args)

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: Dict[str, Any],
    ) -> TensorInverseGamma:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            concentration=attributes["_concentration"],
            rate=attributes["_rate"],
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> TorchInverseGamma:
        """
        Returns the underlying torch.distributions.InverseGamma instance.
        """
        return TorchInverseGamma(
            concentration=self._concentration,
            rate=self._rate,
            validate_args=self._validate_args,
        )

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def concentration(self) -> Tensor:
        """Returns the concentration used to initialize the distribution."""
        return self._concentration

    @property
    def rate(self) -> Tensor:
        """Returns the rate used to initialize the distribution."""
        return self._rate
