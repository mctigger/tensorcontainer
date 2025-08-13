from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.distributions import RelaxedBernoulli as TorchRelaxedBernoulli
from torch.types import Number
from .base import TensorDistribution
from .utils import broadcast_all


class TensorRelaxedBernoulli(TensorDistribution):
    """Tensor-aware RelaxedBernoulli distribution."""

    # Annotated tensor parameters
    _temperature: Tensor
    _probs: Tensor | None = None
    _logits: Tensor | None = None

    def __init__(
        self,
        temperature: Tensor,
        probs: Number | Tensor | None = None,
        logits: Number | Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        self._temperature = temperature

        if probs is not None and logits is not None:
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )

        # broadcast_all is used to lift Number to Tensor
        if probs is not None:
            self._probs, self._temperature = broadcast_all(probs, temperature)
            self._logits = None
        elif logits is not None:
            self._logits, self._temperature = broadcast_all(logits, temperature)
            self._probs = None
        else:
            raise ValueError("Either `probs` or `logits` must be specified.")

        super().__init__(
            self._temperature.shape, self._temperature.device, validate_args
        )

    @classmethod
    def _unflatten_distribution(
        cls, attributes: dict[str, Any]
    ) -> "TensorRelaxedBernoulli":
        """Reconstruct distribution from tensor attributes."""
        return cls(
            temperature=attributes["_temperature"],
            probs=attributes.get("_probs"),
            logits=attributes.get("_logits"),
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> TorchRelaxedBernoulli:
        return TorchRelaxedBernoulli(
            temperature=self._temperature,
            probs=self._probs,
            logits=self._logits,
            validate_args=self._validate_args,
        )

    @property
    def temperature(self) -> Tensor:
        """Returns the temperature used to initialize the distribution."""
        return self.dist().temperature

    @property
    def logits(self) -> Tensor | None:
        """Returns the logits used to initialize the distribution."""
        return self.dist().logits

    @property
    def probs(self) -> Tensor | None:
        """Returns the probabilities used to initialize the distribution."""
        return self.dist().probs
