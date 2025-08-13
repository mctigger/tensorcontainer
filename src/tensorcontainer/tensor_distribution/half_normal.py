from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.distributions import HalfNormal
from .utils import broadcast_all

from .base import TensorDistribution


class TensorHalfNormal(TensorDistribution):
    """Tensor-aware HalfNormal distribution."""

    _scale: Tensor

    def __init__(
        self, scale: Tensor | float, validate_args: bool | None = None
    ) -> None:
        (self._scale,) = broadcast_all(scale)

        shape = self._scale.shape
        device = self._scale.device

        super().__init__(shape, device, validate_args)

    @classmethod
    def _unflatten_distribution(cls, attributes: dict[str, Any]) -> TensorHalfNormal:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            scale=attributes["_scale"],
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> HalfNormal:
        return HalfNormal(scale=self._scale, validate_args=self._validate_args)

    @property
    def scale(self) -> Tensor:
        """Returns the scale used to initialize the distribution."""
        return self.dist().scale
