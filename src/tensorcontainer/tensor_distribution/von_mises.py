from __future__ import annotations

from typing import Any

import torch
from torch.distributions import VonMises as TorchVonMises
from .utils import broadcast_all

from .base import TensorDistribution


class TensorVonMises(TensorDistribution):
    """Tensor-aware VonMises distribution."""

    # Annotated tensor parameters
    _loc: torch.Tensor
    _concentration: torch.Tensor

    def __init__(
        self,
        loc: torch.Tensor,
        concentration: torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self._loc, self._concentration = broadcast_all(loc, concentration)

        super().__init__(self._loc.shape, self._loc.device, validate_args)

    @classmethod
    def _unflatten_distribution(cls, attributes: dict[str, Any]) -> TensorVonMises:
        return cls(
            loc=attributes["_loc"],
            concentration=attributes["_concentration"],
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> TorchVonMises:
        return TorchVonMises(
            loc=self._loc,
            concentration=self._concentration,
            validate_args=self._validate_args,
        )

    @property
    def loc(self) -> torch.Tensor:
        """Returns the loc parameter of the distribution."""
        return self.dist().loc

    @property
    def concentration(self) -> torch.Tensor:
        """Returns the concentration parameter of the distribution."""
        return self.dist().concentration
