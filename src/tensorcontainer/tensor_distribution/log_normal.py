from __future__ import annotations

from typing import Any

from torch import Size, Tensor
from torch.distributions import LogNormal

from .base import TensorDistribution
from .utils import broadcast_all


class TensorLogNormal(TensorDistribution):
    """Tensor-aware LogNormal distribution."""

    # Annotated tensor parameters
    _loc: Tensor
    _scale: Tensor

    def __init__(
        self,
        loc: float | Tensor,
        scale: float | Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self._loc, self._scale = broadcast_all(loc, scale)

        shape = self._loc.shape
        device = self._loc.device

        super().__init__(shape, device, validate_args)

    @classmethod
    def _unflatten_distribution(cls, attributes: dict[str, Any]) -> TensorLogNormal:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            loc=attributes.get("_loc"),  # type: ignore
            scale=attributes.get("_scale"),  # type: ignore
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> LogNormal:
        return LogNormal(
            loc=self._loc, scale=self._scale, validate_args=self._validate_args
        )

    @property
    def loc(self) -> Tensor | None:
        """Returns the loc used to initialize the distribution."""
        return self.dist().loc

    @property
    def scale(self) -> Tensor | None:
        """Returns the scale used to initialize the distribution."""
        return self.dist().scale

    @property
    def param_shape(self) -> Size:
        """Returns the shape of the underlying parameter."""
        return self.batch_shape
