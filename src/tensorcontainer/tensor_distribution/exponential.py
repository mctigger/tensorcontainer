from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.distributions import Exponential as TorchExponential
from .utils import broadcast_all

from .base import TensorDistribution


class TensorExponential(TensorDistribution):
    r"""
    Creates an Exponential distribution parameterized by :attr:`rate`.

    Args:
        rate (float or Tensor): rate = 1 / scale of the distribution
    """

    _rate: Tensor

    def __init__(self, rate: float | Tensor, validate_args: bool | None = None) -> None:
        (self._rate,) = broadcast_all(rate)

        shape = self._rate.shape
        device = self._rate.device

        super().__init__(shape, device, validate_args)

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: dict[str, Any],
    ) -> TensorExponential:
        return cls(
            rate=attributes["_rate"], validate_args=attributes.get("_validate_args")
        )

    def dist(self) -> TorchExponential:
        return TorchExponential(rate=self._rate, validate_args=self._validate_args)

    @property
    def rate(self) -> Tensor:
        return self.dist().rate
