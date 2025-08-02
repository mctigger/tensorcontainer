from __future__ import annotations

from typing import Any, Dict, Optional

from torch import Tensor
from torch.distributions import Distribution
from torch.distributions.utils import broadcast_all

from tensorcontainer.distributions.truncated_normal import TruncatedNormal

from .base import TensorDistribution


class TensorTruncatedNormal(TensorDistribution):
    _loc: Tensor
    _scale: Tensor
    _low: Tensor
    _high: Tensor
    _eps: Tensor
    _validate_args: Optional[bool] = None

    def __init__(
        self,
        loc: Tensor,
        scale: Tensor,
        low: Tensor,
        high: Tensor,
        eps: float = 1e-6,
        validate_args: bool | None = None,
    ):
        loc, scale, low, high, eps = broadcast_all(loc, scale, low, high, eps)
        self._loc = loc
        self._scale = scale
        self._low = low
        self._high = high
        self._eps = eps
        self._validate_args = validate_args

        shape = self._loc.shape
        device = self._loc.device

        super().__init__(shape=shape, device=device, validate_args=validate_args)

    @classmethod
    def _unflatten_distribution(
        cls, attributes: Dict[str, Any]
    ) -> TensorTruncatedNormal:
        instance = cls(
            loc=attributes["_loc"],
            scale=attributes["_scale"],
            low=attributes["_low"],
            high=attributes["_high"],
            eps=attributes["_eps"],
            validate_args=attributes.get("_validate_args"),
        )
        return instance

    @property
    def loc(self) -> Tensor:
        return self._loc

    @property
    def scale(self) -> Tensor:
        return self._scale

    @property
    def low(self) -> Tensor:
        return self._low

    @property
    def high(self) -> Tensor:
        return self._high

    def dist(self) -> Distribution:
        return TruncatedNormal(
            self._loc,
            self._scale,
            self._low,
            self._high,
            eps=self._eps,
            validate_args=self._validate_args,
        )
