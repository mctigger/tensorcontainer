from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.distributions import Weibull as TorchWeibull


from .base import TensorDistribution
from .utils import broadcast_all


class TensorWeibull(TensorDistribution):
    """Tensor-aware Weibull distribution."""

    _scale: Tensor
    _concentration: Tensor

    def __init__(
        self,
        scale: float | Tensor,
        concentration: float | Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self._scale, self._concentration = broadcast_all(scale, concentration)

        shape = self._scale.shape
        device = self._scale.device

        super().__init__(shape, device, validate_args)

    @classmethod
    def _unflatten_distribution(cls, attributes: dict[str, Any]) -> TensorWeibull:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            scale=attributes["_scale"],
            concentration=attributes["_concentration"],
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> TorchWeibull:
        return TorchWeibull(
            scale=self._scale,
            concentration=self._concentration,
            validate_args=self._validate_args,
        )

    @property
    def scale(self) -> Tensor:
        """Returns the scale used to initialize the distribution."""
        return self._scale

    @property
    def concentration(self) -> Tensor:
        """Returns the concentration used to initialize the distribution."""
        return self._concentration
