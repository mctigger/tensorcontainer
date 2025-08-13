from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.distributions import Uniform

from .base import TensorDistribution
from .utils import broadcast_all


class TensorUniform(TensorDistribution):
    _low: Tensor
    _high: Tensor

    def __init__(
        self,
        low: float | Tensor,
        high: float | Tensor,
        validate_args: bool | None = None,
    ) -> None:
        low, high = broadcast_all(low, high)
        self._low = low
        self._high = high

        super().__init__(low.shape, low.device, validate_args)

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: dict[str, Any],
    ) -> TensorUniform:
        return cls(
            low=attributes["_low"],
            high=attributes["_high"],
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> Uniform:
        return Uniform(
            low=self._low, high=self._high, validate_args=self._validate_args
        )

    @property
    def low(self) -> Tensor:
        return self.dist().low

    @property
    def high(self) -> Tensor:
        return self.dist().high
