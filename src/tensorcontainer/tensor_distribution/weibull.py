from __future__ import annotations

from typing import Any, Dict

from torch import Tensor
from torch.distributions import Weibull as TorchWeibull

from tensorcontainer.tensor_annotated import TDCompatible

from .base import TensorDistribution


class TensorWeibull(TensorDistribution):
    """Tensor-aware Weibull distribution."""

    # Annotated tensor parameters
    _scale: Tensor
    _concentration: Tensor

    def __init__(self, scale: Tensor, concentration: Tensor):
        # Parameter validation
        if scale is None or scale.numel() == 0:
            raise ValueError("`scale` must be provided and non-empty.")
        if concentration is None or concentration.numel() == 0:
            raise ValueError("`concentration` must be provided and non-empty.")

        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        self._scale = scale
        self._concentration = concentration

        shape = scale.shape
        device = scale.device

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        tensor_attributes: Dict[str, TDCompatible],
        meta_attributes: Dict[str, Any],
    ) -> "TensorWeibull":
        """Reconstruct distribution from tensor attributes."""
        return cls(
            scale=tensor_attributes["_scale"],  # type: ignore
            concentration=tensor_attributes["_concentration"],  # type: ignore
        )

    def dist(self) -> TorchWeibull:
        return TorchWeibull(scale=self._scale, concentration=self._concentration)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def scale(self) -> Tensor:
        """Returns the scale used to initialize the distribution."""
        return self.dist().scale

    @property
    def concentration(self) -> Tensor:
        """Returns the concentration used to initialize the distribution."""
        return self.dist().concentration
