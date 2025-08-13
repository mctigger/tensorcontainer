from __future__ import annotations

from typing import Any

from torch import Size, Tensor
from torch.distributions import (
    Categorical,
)

from .base import TensorDistribution


class TensorCategorical(TensorDistribution):
    """Tensor-aware categorical distribution using Categorical."""

    # Annotated tensor parameters
    _probs: Tensor | None = None
    _logits: Tensor | None = None

    def __init__(
        self,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        data = probs if probs is not None else logits
        # Parameter validation occurs in super().__init__(), but we need an early
        # check here to safely derive shape and device from the data tensor
        # before calling the parent constructor
        if data is None:
            raise ValueError("Either 'probs' or 'logits' must be provided.")

        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        self._probs = probs
        self._logits = logits

        # For categorical distributions the last dimension must not be part of the
        # shape since it contains the probabilities for each class and thus, should
        # never change.
        shape = data.shape[:-1]
        device = data.device

        # The batch shape is all dimensions except the last one.
        super().__init__(shape, device, validate_args)

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: dict[str, Any],
    ) -> TensorCategorical:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            probs=attributes.get("_probs"),  # type: ignore
            logits=attributes.get("_logits"),  # type: ignore
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> Categorical:
        return Categorical(
            probs=self._probs, logits=self._logits, validate_args=self._validate_args
        )

    @property
    def logits(self) -> Tensor:
        """Returns the logits used to initialize the distribution."""
        return self.dist().logits

    @property
    def probs(self) -> Tensor | None:
        """Returns the probabilities used to initialize the distribution."""
        return self.dist().probs

    @property
    def param_shape(self) -> Size:
        """Returns the shape of the underlying parameter."""
        return self.dist().param_shape
