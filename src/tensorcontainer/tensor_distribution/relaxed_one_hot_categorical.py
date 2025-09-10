from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.distributions import RelaxedOneHotCategorical

from .base import TensorDistribution


class TensorRelaxedOneHotCategorical(TensorDistribution):
    """Tensor-aware RelaxedCategorical distribution."""

    _temperature: tuple[Tensor, ...]
    _probs: Tensor | None = None
    _logits: Tensor | None = None

    def __init__(
        self,
        temperature: Tensor,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        """
        There is a bug in RelaxedOneHotCategorical https://github.com/pytorch/pytorch/issues/37162
        That is why we only allowed scalar temperatures for now.
        """
        if temperature.ndim > 0:
            raise ValueError(
                "Expected scalar temperature tensor. This is because of a bug in torch: https://github.com/pytorch/pytorch/issues/37162"
            )

        data = probs if probs is not None else logits
        if data is None:
            raise ValueError("Either 'probs' or 'logits' must be provided.")

        # Determine shape and device from data (probs or logits)
        shape = data.shape[:-1]
        device = data.device

        # Use tuple such that we can annotate it for flatten / unflatten, but it is handled as metadata
        self._temperature = (temperature,)
        self._probs = probs
        self._logits = logits

        super().__init__(shape, device, validate_args)

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: dict[str, Any],
    ) -> TensorRelaxedOneHotCategorical:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            temperature=attributes["_temperature"][0],
            probs=attributes.get("_probs"),
            logits=attributes.get("_logits"),
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> RelaxedOneHotCategorical:
        return RelaxedOneHotCategorical(
            temperature=self._temperature[0],
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
