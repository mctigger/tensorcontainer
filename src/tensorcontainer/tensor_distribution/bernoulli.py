from __future__ import annotations

from typing import Any

from torch import Size, Tensor
from torch.distributions import Bernoulli
from .utils import broadcast_all
from torch.types import Number

from .base import TensorDistribution


class TensorBernoulli(TensorDistribution):
    """Tensor-aware Bernoulli distribution."""

    # Annotated tensor parameters
    _probs: Tensor | None
    _logits: Tensor | None

    def __init__(
        self,
        probs: Number | Tensor | None = None,
        logits: Number | Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )

        # broadcast_all is used to lift Number to Tensor
        if probs is not None:
            (self._probs,) = broadcast_all(probs)
            shape = self._probs.shape
            device = self._probs.device
            self._logits = None
        else:
            (self._logits,) = broadcast_all(logits)
            shape = self._logits.shape
            device = self._logits.device
            self._probs = None

        super().__init__(shape, device, validate_args)

    @classmethod
    def _unflatten_distribution(cls, attributes: dict[str, Any]) -> TensorBernoulli:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            probs=attributes.get("_probs"),
            logits=attributes.get("_logits"),
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> Bernoulli:
        return Bernoulli(
            probs=self._probs, logits=self._logits, validate_args=self._validate_args
        )

    @property
    def logits(self) -> Tensor:
        return self.dist().logits

    @property
    def probs(self) -> Tensor:
        return self.dist().probs

    @property
    def param_shape(self) -> Size:
        return self.dist().param_shape
