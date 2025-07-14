from __future__ import annotations

from typing import Any, Dict, Union

from torch import Tensor
from torch.distributions import Beta
from torch.distributions.utils import broadcast_all

from .base import TensorDistribution


class TensorBeta(TensorDistribution):
    """Tensor-aware Beta distribution."""

    # Annotated tensor parameters
    _concentration1: Union[Tensor, float]
    _concentration0: Union[Tensor, float]

    def __init__(
        self, concentration1: Union[Tensor, float], concentration0: Union[Tensor, float]
    ):
        # Broadcast parameters into matching shape and devices
        concentration1, concentration0 = broadcast_all(concentration1, concentration0)

        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        self._concentration1 = concentration1
        self._concentration0 = concentration0

        super().__init__(concentration1.shape, concentration1.device)

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: Dict[str, Any],
    ) -> TensorBeta:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            concentration1=attributes.get("_concentration1"),  # type: ignore
            concentration0=attributes.get("_concentration0"),  # type: ignore
        )

    def dist(self) -> Beta:
        return Beta(
            concentration1=self._concentration1, concentration0=self._concentration0
        )

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def concentration1(self) -> Tensor:
        """Returns the concentration1 parameter of the distribution."""
        return self.dist().concentration1

    @property
    def concentration0(self) -> Tensor:
        """Returns the concentration0 parameter of the distribution."""
        return self.dist().concentration0

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the Beta distribution."""
        return self.dist().variance

    @property
    def mean(self) -> Tensor:
        """Returns the mean of the Beta distribution."""
        return self.dist().mean
