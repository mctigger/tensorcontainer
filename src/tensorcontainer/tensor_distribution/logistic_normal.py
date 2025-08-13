from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.distributions import LogisticNormal as TorchLogisticNormal
from torch.distributions.distribution import Distribution

from .base import TensorDistribution
from .utils import broadcast_all


class TensorLogisticNormal(TensorDistribution):
    _loc: Tensor
    _scale: Tensor

    def __init__(
        self, loc: Tensor, scale: Tensor, validate_args: bool | None = None
    ) -> None:
        loc, scale = broadcast_all(loc, scale)

        if loc.ndim == 0:
            loc = loc.expand(1)
            scale = scale.expand(1)

        self._loc = loc
        self._scale = scale

        shape = loc.shape
        device = loc.device

        super().__init__(shape, device, validate_args)

    def dist(self) -> Distribution:
        return TorchLogisticNormal(
            loc=self._loc, scale=self._scale, validate_args=self._validate_args
        )

    @property
    def loc(self) -> Tensor:
        """Returns the location parameter of the distribution."""
        return self._loc

    @property
    def scale(self) -> Tensor:
        """Returns the scale parameter of the distribution."""
        return self._scale

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: dict[str, Any],
    ) -> TensorLogisticNormal:
        return cls(
            loc=attributes["_loc"],
            scale=attributes["_scale"],
            validate_args=attributes.get("_validate_args"),
        )
