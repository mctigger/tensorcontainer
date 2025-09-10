from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.distributions import Kumaraswamy as TorchKumaraswamy
from .utils import broadcast_all

from .base import TensorDistribution


class TensorKumaraswamy(TensorDistribution):
    """Tensor-aware Kumaraswamy distribution."""

    # Annotated tensor parameters
    _concentration1: Tensor
    _concentration0: Tensor

    def __init__(
        self,
        concentration1: float | Tensor,
        concentration0: float | Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self._concentration1, self._concentration0 = broadcast_all(
            concentration1, concentration0
        )

        shape = self._concentration1.shape
        device = self._concentration1.device

        super().__init__(shape, device, validate_args)

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: dict[str, Any],
    ) -> TensorKumaraswamy:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            concentration1=attributes["_concentration1"],
            concentration0=attributes["_concentration0"],
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> TorchKumaraswamy:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return TorchKumaraswamy(
            concentration1=self._concentration1,
            concentration0=self._concentration0,
            validate_args=self._validate_args,
        )

    @property
    def concentration1(self) -> Tensor:
        """Returns the concentration1 parameter of the distribution."""
        return self.dist().concentration1

    @property
    def concentration0(self) -> Tensor:
        """Returns the concentration0 parameter of the distribution."""
        return self.dist().concentration0
