from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.distributions import Distribution
from torch.types import Number

from .base import TensorDistribution
from .utils import broadcast_all


class TensorSymLog(TensorDistribution):
    """Tensor-aware SymLog distribution.

    Creates a SymLog distribution parameterized by `loc` (mean) and `scale` (standard deviation).
    This distribution transforms a Normal distribution with a symexp transform, which is useful
    for modeling data with a wide dynamic range where the data can be both positive and negative.

    Args:
        loc: Mean of the base Normal distribution.
        scale: Standard deviation of the base Normal distribution. Must be positive.
        validate_args: Whether to validate the arguments. Defaults to None.

    Note:
        The SymLog distribution is useful for modeling data with a wide dynamic range,
        where the data can be both positive and negative, and can have values close to zero.
        The symlog transform compresses large values and expands small values, making the
        distribution more stable for optimization.
    """

    # Annotated tensor parameters
    _loc: Tensor
    _scale: Tensor

    def __init__(
        self,
        loc: Number | Tensor,
        scale: Number | Tensor,
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
    ) -> TensorSymLog:
        return cls(
            loc=attributes["_loc"],
            scale=attributes["_scale"],
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> Distribution:
        """Return the underlying SymLogDistribution instance."""
        from tensorcontainer.distributions.symlog import SymLogDistribution

        return SymLogDistribution(
            loc=self._loc,
            scale=self._scale,
            validate_args=self._validate_args,
        )

    @property
    def loc(self) -> Tensor:
        """Returns the location parameter of the distribution."""
        return self._loc

    @property
    def scale(self) -> Tensor:
        """Returns the scale parameter of the distribution."""
        return self._scale
