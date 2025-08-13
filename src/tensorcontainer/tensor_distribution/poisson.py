from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.distributions import Poisson

from .base import TensorDistribution
from torch.types import Number
from .utils import broadcast_all


class TensorPoisson(TensorDistribution):
    _rate: Tensor

    def __init__(
        self, rate: Number | Tensor, validate_args: bool | None = None
    ) -> None:
        (self._rate,) = broadcast_all(rate)

        shape = self._rate.shape
        device = self._rate.device

        super().__init__(shape, device, validate_args)

    @classmethod
    def _unflatten_distribution(cls, attributes: dict[str, Any]) -> TensorPoisson:
        return cls(
            rate=attributes["_rate"],
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> Poisson:
        return Poisson(rate=self._rate, validate_args=self._validate_args)

    @property
    def rate(self) -> Tensor:
        return self.dist().rate
