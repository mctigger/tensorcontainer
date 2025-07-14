from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Size, Tensor
from torch.distributions import Distribution, Gamma

from tensorcontainer.tensor_annotated import TDCompatible

from .base import TensorDistribution


class TensorGamma(TensorDistribution):
    """Tensor-aware Gamma distribution."""

    # Annotated tensor parameters
    _concentration: Optional[Tensor] = None
    _rate: Optional[Tensor] = None

    def __init__(self, concentration: Tensor, rate: Tensor):
        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        self._concentration = concentration
        self._rate = rate

        # Determine batch_shape and device
        batch_shape = torch.broadcast_shapes(concentration.shape, rate.shape)
        device = concentration.device if concentration.is_cuda else rate.device

        super().__init__(shape=batch_shape, device=device)

    @classmethod
    def _unflatten_distribution(
        cls,
        tensor_attributes: Dict[str, TDCompatible],
        meta_attributes: Dict[str, Any],
    ) -> TensorGamma:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            concentration=tensor_attributes["_concentration"].as_tensor(), # type: ignore
            rate=tensor_attributes["_rate"].as_tensor(), # type: ignore
        )

    def dist(self) -> Distribution:
        # The constructor ensures concentration and rate are Tensors, so no None check needed here.
        return Gamma(concentration=self.concentration, rate=self.rate)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def concentration(self) -> Tensor:
        """Returns the concentration used to initialize the distribution."""
        return self._concentration # type: ignore

    @property
    def rate(self) -> Tensor:
        """Returns the rate used to initialize the distribution."""
        return self._rate # type: ignore

    @property
    def mean(self) -> Tensor:
        return self.dist().mean

    @property
    def variance(self) -> Tensor:
        return self.dist().variance

    @property
    def stddev(self) -> Tensor:
        return self.dist().stddev

    @property
    def param_shape(self) -> Size:
        """Returns the shape of the underlying parameter."""
        # This property is not directly available in torch.distributions.Gamma,
        # but it's common in other TensorDistribution subclasses.
        # We can derive it from the concentration or rate.
        if self._concentration is not None:
            return self._concentration.shape
        elif self._rate is not None:
            return self._rate.shape
        else:
            raise RuntimeError("Neither concentration nor rate is set.")
