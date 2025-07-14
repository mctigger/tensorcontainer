from __future__ import annotations

from typing import Any, Dict, Optional

import torch  # Add this import
from torch import Size, Tensor
from torch.distributions import Multinomial

from tensorcontainer.tensor_annotated import TDCompatible

from .base import TensorDistribution


class TensorMultinomial(TensorDistribution):
    """Tensor-aware Multinomial distribution."""

    # Annotated tensor parameters
    _total_count: Tensor
    _probs: Optional[Tensor] = None
    _logits: Optional[Tensor] = None

    def __init__(
        self, total_count: int = 1, probs: Optional[Tensor] = None, logits: Optional[Tensor] = None
    ):
        # Parameter validation
        if not isinstance(total_count, int) or total_count < 0:
            raise ValueError("total_count must be a non-negative integer.")

        if probs is None and logits is None:
            raise RuntimeError("Either 'probs' or 'logits' must be provided.")
        if probs is not None and logits is not None:
            raise RuntimeError("Only one of 'probs' or 'logits' can be provided.")

        data = probs if probs is not None else logits
        if data is None: # This case is already handled by the above checks, but for mypy
            raise RuntimeError("Internal error: data tensor is None.")

        # Store the parameters in annotated attributes before calling super().__init__()
        # Convert total_count to a scalar Tensor
        self._total_count = torch.tensor(total_count, dtype=torch.float, device=data.device)
        self._probs = probs
        self._logits = logits

        # The batch shape is all dimensions except the last one.
        shape = data.shape[:-1]
        device = data.device

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        tensor_attributes: Dict[str, TDCompatible],
        meta_attributes: Dict[str, Any],
    ) -> TensorMultinomial:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            total_count=tensor_attributes["_total_count"],  # type: ignore
            probs=tensor_attributes.get("_probs"),  # type: ignore
            logits=tensor_attributes.get("_logits"),  # type: ignore
        )

    def dist(self) -> Multinomial: # Changed return type
        return Multinomial( # Removed Independent wrapper
            total_count=int(self._total_count.item()), probs=self._probs, logits=self._logits
        )

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def total_count(self) -> Tensor:
        """Returns the total_count used to initialize the distribution."""
        return self._total_count

    @property
    def logits(self) -> Optional[Tensor]:
        """Returns the logits used to initialize the distribution."""
        return self.dist().logits # Access directly

    @property
    def probs(self) -> Optional[Tensor]:
        """Returns the probabilities used to initialize the distribution."""
        return self.dist().probs # Access directly
    @property
    def param_shape(self) -> Size:
        """Returns the shape of the underlying parameter."""
        # The param_shape should be the shape of the probs/logits tensor
        # including the last dimension (number of categories)
        if self._probs is not None:
            return self._probs.shape
        elif self._logits is not None:
            return self._logits.shape
        else:
            raise RuntimeError("Neither probs nor logits are available.")
