from __future__ import annotations

from typing import Any

from torch import Size, Tensor
from torch.distributions import Multinomial

from .base import TensorDistribution


class TensorMultinomial(TensorDistribution):
    """Tensor-aware Multinomial distribution."""

    # Annotated tensor parameters
    _total_count: int
    _probs: Tensor | None = None
    _logits: Tensor | None = None

    def __init__(
        self,
        total_count: int = 1,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        data = probs if probs is not None else logits
        if data is None:
            raise ValueError("Either 'probs' or 'logits' must be provided.")

        self._total_count = total_count
        self._probs = probs
        self._logits = logits

        # The batch shape is all dimensions except the last one.
        shape = data.shape[:-1]
        device = data.device

        super().__init__(shape, device, validate_args)

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: dict[str, Any],
    ) -> TensorMultinomial:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            total_count=attributes["_total_count"],
            probs=attributes.get("_probs"),
            logits=attributes.get("_logits"),
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> Multinomial:
        return Multinomial(
            total_count=self._total_count,
            probs=self._probs,
            logits=self._logits,
            validate_args=self._validate_args,
        )

    @property
    def total_count(self) -> int:
        """Returns the total_count used to initialize the distribution."""
        return self._total_count

    @property
    def logits(self) -> Tensor | None:
        """Returns the logits used to initialize the distribution."""
        return self.dist().logits  # Access directly

    @property
    def probs(self) -> Tensor | None:
        """Returns the probabilities used to initialize the distribution."""
        return self.dist().probs  # Access directly

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
            raise ValueError("Neither probs nor logits are available.")
