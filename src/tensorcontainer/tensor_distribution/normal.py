from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.distributions import Normal
from .utils import broadcast_all

from .base import TensorDistribution


class TensorNormal(TensorDistribution):
    """Tensor-aware Normal distribution.

    Creates a Normal distribution parameterized by `loc` (mean) and `scale` (standard deviation).

    Args:
        loc: Mean of the distribution.
        scale: Standard deviation of the distribution. Must be positive.

    Note:
        The Normal distribution is also known as the Gaussian distribution.
    """

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
    def _unflatten_distribution(
        cls,
        attributes: dict[str, Any],
    ) -> TensorNormal:
        return cls(
            loc=attributes["_loc"],
            scale=attributes["_scale"],
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> Normal:
        return Normal(
            loc=self._loc,
            scale=self._scale,
            validate_args=self._validate_args,
        )

    @property
    def loc(self) -> Tensor:
        """Returns the location parameter of the distribution."""
        return self.dist().loc

    @property
    def scale(self) -> Tensor:
        """Returns the scale parameter of the distribution."""
        return self.dist().scale
