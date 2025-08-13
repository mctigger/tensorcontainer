from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.distributions import Laplace
from .utils import broadcast_all

from .base import TensorDistribution


class TensorLaplace(TensorDistribution):
    """Tensor-aware Laplace distribution."""

    # Annotated tensor parameters
    _loc: Tensor
    _scale: Tensor

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self._loc, self._scale = broadcast_all(loc, scale)
        shape = self._loc.shape
        device = self._loc.device
        super().__init__(shape, device, validate_args)

    @classmethod
    def _unflatten_distribution(cls, attributes: dict[str, Any]) -> TensorLaplace:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            loc=attributes["_loc"],
            scale=attributes["_scale"],
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> Laplace:
        return Laplace(
            loc=self._loc,
            scale=self._scale,
            validate_args=self._validate_args,
        )

    @property
    def loc(self) -> Tensor | None:
        """Returns the loc used to initialize the distribution."""
        return self.dist().loc

    @property
    def scale(self) -> Tensor | None:
        """Returns the scale used to initialize the distribution."""
        return self.dist().scale
