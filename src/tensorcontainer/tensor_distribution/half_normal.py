from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.distributions import HalfNormal

from tensorcontainer.tensor_annotated import TDCompatible

from .base import TensorDistribution


class TensorHalfNormal(TensorDistribution):
    """Tensor-aware HalfNormal distribution."""

    # Annotated tensor parameters
    _scale: Optional[Tensor] = None

    def __init__(self, scale: Tensor):
        if torch.any(scale <= 0):
            raise ValueError("scale must be positive")

        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        self._scale = scale

        shape = scale.shape
        device = scale.device

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        tensor_attributes: Dict[str, TDCompatible],
        meta_attributes: Dict[str, Any],
    ) -> TensorHalfNormal:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            scale=tensor_attributes.get("_scale"),  # type: ignore
        )

    def dist(self) -> HalfNormal:
        return HalfNormal(scale=self._scale)  # type: ignore

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the HalfNormal distribution."""
        return self.dist().variance

    @property
    def scale(self) -> Tensor:
        """Returns the scale used to initialize the distribution."""
        return self.dist().scale

