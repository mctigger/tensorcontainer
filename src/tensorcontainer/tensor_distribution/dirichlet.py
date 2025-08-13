from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.distributions import Dirichlet

from .base import TensorDistribution


class TensorDirichlet(TensorDistribution):
    """Tensor-aware Dirichlet distribution."""

    # Annotated tensor parameters
    _concentration: Tensor

    def __init__(
        self, concentration: Tensor, validate_args: bool | None = None
    ) -> None:
        self._concentration = concentration
        super().__init__(concentration.shape, concentration.device, validate_args)

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: dict[str, Any],
    ) -> TensorDirichlet:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            concentration=attributes["_concentration"],
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> Dirichlet:
        return Dirichlet(
            concentration=self._concentration, validate_args=self._validate_args
        )

    @property
    def concentration(self) -> Tensor:
        return self.dist().concentration
