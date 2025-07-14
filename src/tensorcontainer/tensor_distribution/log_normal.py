from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Size, Tensor
from torch.distributions import LogNormal

from tensorcontainer.tensor_annotated import TDCompatible

from .base import TensorDistribution


class TensorLogNormal(TensorDistribution):
    """Tensor-aware LogNormal distribution."""

    # Annotated tensor parameters
    _loc: Optional[Tensor] = None
    _scale: Optional[Tensor] = None

    def __init__(self, loc: Tensor, scale: Tensor):
        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        self._loc = loc
        self._scale = scale

        if torch.any(scale <= 0):
            raise ValueError("scale must be positive")

        try:
            data = torch.broadcast_tensors(loc, scale)
        except RuntimeError as e:
            raise ValueError(f"loc and scale must have compatible shapes: {e}")

        shape = data[0].shape
        device = data[0].device

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        tensor_attributes: Dict[str, TDCompatible],
        meta_attributes: Dict[str, Any],
    ) -> TensorLogNormal:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            loc=tensor_attributes.get("_loc"),  # type: ignore
            scale=tensor_attributes.get("_scale"),  # type: ignore
        )

    def dist(self) -> LogNormal:
        return LogNormal(loc=self._loc, scale=self._scale)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def loc(self) -> Optional[Tensor]:
        """Returns the loc used to initialize the distribution."""
        return self.dist().loc

    @property
    def scale(self) -> Optional[Tensor]:
        """Returns the scale used to initialize the distribution."""
        return self.dist().scale

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the LogNormal distribution."""
        return self.dist().variance

    @property
    def param_shape(self) -> Size:
        """Returns the shape of the underlying parameter."""
        return self.batch_shape
