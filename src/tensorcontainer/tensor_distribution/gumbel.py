from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.distributions import Gumbel as TorchGumbel
from .utils import broadcast_all

from .base import TensorDistribution


class TensorGumbel(TensorDistribution):
    """
    A Gumbel distribution.

    This distribution is parameterized by `loc` and `scale`.

    Source: https://pytorch.org/docs/stable/distributions.html#gumbel
    """

    _loc: Tensor
    _scale: Tensor

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self._loc, self._scale = broadcast_all(loc, scale)
        super().__init__(self._loc.shape, self._loc.device, validate_args)

    @classmethod
    def _unflatten_distribution(cls, attributes: dict[str, Any]) -> TensorGumbel:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            loc=attributes["_loc"],
            scale=attributes["_scale"],
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> TorchGumbel:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return TorchGumbel(
            loc=self._loc,
            scale=self._scale,
            validate_args=self._validate_args,
        )

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def loc(self) -> Tensor:
        """Returns the loc used to initialize the distribution."""
        return self._loc

    @property
    def scale(self) -> Tensor:
        """Returns the scale used to initialize the distribution."""
        return self._scale
