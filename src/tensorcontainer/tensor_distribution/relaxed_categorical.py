from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.distributions import RelaxedOneHotCategorical as TorchRelaxedCategorical


from .base import TensorDistribution


class TensorRelaxedCategorical(TensorDistribution):
    """Tensor-aware RelaxedCategorical distribution."""

    # Annotated tensor parameters
    _temperature: Tensor
    _probs: Optional[Tensor] = None
    _logits: Optional[Tensor] = None

    def __init__(
        self,
        temperature: Optional[Tensor],
        probs: Optional[Tensor] = None,
        logits: Optional[Tensor] = None,
    ):
        data = probs if probs is not None else logits
        if data is None:
            raise RuntimeError("Either 'probs' or 'logits' must be provided.")
        if temperature is None:
            raise RuntimeError("'temperature' must be provided.")

        # Convert temperature to a tensor if it's a float or int
        if isinstance(temperature, (float, int)):
            temperature = torch.tensor(
                temperature, dtype=data.dtype, device=data.device
            )

        # After this point, temperature is guaranteed to be a Tensor due to the checks above
        # and the initial type hint being Optional[Tensor]
        assert temperature is not None  # For mypy

        # Determine shape and device from data (probs or logits)
        shape = data.shape[:-1]
        device = data.device

        self._temperature = temperature
        self._probs = probs
        self._logits = logits

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: Dict[str, Any],
    ) -> "TensorRelaxedCategorical":
        """Reconstruct distribution from tensor attributes."""
        return cls(
            temperature=attributes["_temperature"],  # type: ignore
            probs=attributes.get("_probs"),  # type: ignore
            logits=attributes.get("_logits"),  # type: ignore
        )

    def dist(self) -> TorchRelaxedCategorical:
        return TorchRelaxedCategorical(
            temperature=self._temperature,
            probs=self._probs,
            logits=self._logits,
        )

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def temperature(self) -> Tensor:
        """Returns the temperature used to initialize the distribution."""
        return self.dist().temperature

    @property
    def logits(self) -> Optional[Tensor]:
        """Returns the logits used to initialize the distribution."""
        return self.dist().logits

    @property
    def probs(self) -> Optional[Tensor]:
        """Returns the probabilities used to initialize the distribution."""
        return self.dist().probs
