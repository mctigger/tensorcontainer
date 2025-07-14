from __future__ import annotations

from typing import Any, Dict

from torch import Size, Tensor
from torch.distributions import HalfCauchy as TorchHalfCauchy

from tensorcontainer.tensor_annotated import TDCompatible

from .base import TensorDistribution


class TensorHalfCauchy(TensorDistribution):
    """Tensor-aware HalfCauchy distribution."""

    # Annotated tensor parameters
    _scale: Tensor

    def __init__(self, scale: Tensor):
        # Parameter validation occurs in super().__init__(), but we need an early
        # check here to safely derive shape and device from the data tensor
        # before calling the parent constructor
        if scale is None:
            raise RuntimeError("`scale` must be provided.")

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
    ) -> TensorHalfCauchy:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            scale=tensor_attributes["_scale"],
        )

    def dist(self) -> TorchHalfCauchy:
        return TorchHalfCauchy(scale=self._scale)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def scale(self) -> Tensor:
        """Returns the scale used to initialize the distribution."""
        return self.dist().scale

    @property
    def param_shape(self) -> Size:
        """Returns the shape of the underlying parameter."""
        return self.dist().param_shape
