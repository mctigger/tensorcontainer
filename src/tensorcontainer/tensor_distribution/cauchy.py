from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.distributions import Cauchy
from .utils import broadcast_all

from .base import TensorDistribution


class TensorCauchy(TensorDistribution):
    """Tensor-aware Cauchy distribution.

    Creates a Cauchy distribution parameterized by `loc` (location) and `scale` parameters.
    The Cauchy distribution is a continuous probability distribution with heavy tails.

    Args:
        loc (float or Tensor): mode or median of the distribution.
        scale (float or Tensor): half width at half maximum.

    Note:
        The Cauchy distribution has no finite mean or variance. These properties
        are not implemented as they would return undefined values.
    """

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
    def _unflatten_distribution(cls, attributes: dict[str, Any]) -> TensorCauchy:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            loc=attributes["_loc"],
            scale=attributes["_scale"],
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> Cauchy:
        return Cauchy(
            loc=self._loc, scale=self._scale, validate_args=self._validate_args
        )

    @property
    def loc(self) -> Tensor:
        """Returns the location parameter of the distribution."""
        return self.dist().loc

    @property
    def scale(self) -> Tensor:
        """Returns the scale parameter of the distribution."""
        return self.dist().scale
