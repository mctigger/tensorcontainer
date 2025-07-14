from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Size, Tensor
from torch.distributions import NegativeBinomial

from tensorcontainer.tensor_annotated import TDCompatible

from .base import TensorDistribution


class TensorNegativeBinomial(TensorDistribution):
    """Tensor-aware NegativeBinomial distribution."""

    # Annotated tensor parameters
    _total_count: Tensor
    _probs: Optional[Tensor] = None
    _logits: Optional[Tensor] = None

    def __init__(
        self,
        total_count: Tensor,
        probs: Optional[Tensor] = None,
        logits: Optional[Tensor] = None,
    ):
        # Parameter validation
        if torch.any(total_count <= 0):
            raise ValueError("total_count must be positive")

        if probs is None and logits is None:
            raise ValueError("Either 'probs' or 'logits' must be provided.")
        if probs is not None and logits is not None:
            raise ValueError("Only one of 'probs' or 'logits' can be specified.")

        param = probs if probs is not None else logits
        # Assert param is not None after validation
        assert param is not None

        try:
            torch.broadcast_tensors(total_count, param)
        except RuntimeError as e:
            raise ValueError(
                f"total_count and probs/logits must have compatible shapes: {e}"
            )

        # Store the parameters in annotated attributes before calling super().__init__()
        self._total_count = total_count
        self._probs = probs
        self._logits = logits

        # Derive shape and device from the parameters
        shape = torch.broadcast_shapes(total_count.shape, param.shape)
        device = total_count.device

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        tensor_attributes: Dict[str, TDCompatible],
        meta_attributes: Dict[str, Any],
    ) -> TensorNegativeBinomial:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            total_count=tensor_attributes["_total_count"],  # type: ignore
            probs=tensor_attributes.get("_probs"),  # type: ignore
            logits=tensor_attributes.get("_logits"),  # type: ignore
        )

    def dist(self) -> NegativeBinomial:
        return NegativeBinomial(
            total_count=self._total_count, probs=self._probs, logits=self._logits
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
        return self.dist().logits

    @property
    def probs(self) -> Optional[Tensor]:
        """Returns the probabilities used to initialize the distribution."""
        return self.dist().probs

    @property
    def param_shape(self) -> Size:
        """Returns the shape of the underlying parameter."""
        return self.dist().param_shape
