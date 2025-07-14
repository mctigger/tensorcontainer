from __future__ import annotations

from typing import Any, Dict, Optional

from torch import Size, Tensor
from torch.distributions import (
    OneHotCategoricalStraightThrough,
)

from tensorcontainer.tensor_annotated import TDCompatible

from .base import TensorDistribution


class TensorCategorical(TensorDistribution):
    """Tensor-aware categorical distribution using OneHotCategoricalStraightThrough."""
    
    # Annotated tensor parameters
    _probs: Optional[Tensor] = None
    _logits: Optional[Tensor] = None

    def __init__(self, probs: Optional[Tensor] = None, logits: Optional[Tensor] = None):
        data = probs if probs is not None else logits
        # Parameter validation occurs in super().__init__(), but we need an early
        # check here to safely derive shape and device from the data tensor
        # before calling the parent constructor
        if data is None:
            raise RuntimeError("Either 'probs' or 'logits' must be provided.")
        
        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        self._probs = probs
        self._logits = logits
        
        shape = data.shape[:-1]
        device = data.device

        # The batch shape is all dimensions except the last one.
        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        tensor_attributes: Dict[str, TDCompatible],
        meta_attributes: Dict[str, Any],
    ) -> TensorCategorical:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            probs=tensor_attributes.get("_probs"),  # type: ignore
            logits=tensor_attributes.get("_logits"),  # type: ignore
        )

    def dist(self) -> OneHotCategoricalStraightThrough:
        return OneHotCategoricalStraightThrough(
            probs=self._probs, logits=self._logits
        )

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def logits(self) -> Optional[Tensor]:
        """Returns the logits used to initialize the distribution."""
        return self.dist().logits

    @property
    def probs(self) -> Optional[Tensor]:
        """Returns the probabilities used to initialize the distribution."""
        return self.dist().probs

    @property
    def param_shape(self) -> Size:
        """Returns the shape of the underlying parameter."""
        return self.dist().param_shape
