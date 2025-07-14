from __future__ import annotations

from typing import Any, Dict, Optional

from torch import Tensor
from torch.distributions import RelaxedBernoulli as TorchRelaxedBernoulli

from tensorcontainer.tensor_annotated import TDCompatible

from .base import TensorDistribution


class TensorRelaxedBernoulli(TensorDistribution):
    """Tensor-aware RelaxedBernoulli distribution."""

    # Annotated tensor parameters
    _temperature: Tensor
    _probs: Optional[Tensor] = None
    _logits: Optional[Tensor] = None

    def __init__(
        self,
        temperature: Tensor,
        probs: Optional[Tensor] = None,
        logits: Optional[Tensor] = None,
    ):
        # Parameter validation occurs in super().__init__(), but we need an early
        # check here to safely derive shape and device from the data tensor
        # before calling the parent constructor
        if temperature is None:
            raise RuntimeError("`temperature` must be provided.")
        data = probs if probs is not None else logits
        if data is None:
            raise RuntimeError("Either `probs` or `logits` must be provided.")

        self._temperature = temperature
        self._probs = probs
        self._logits = logits

        shape = data.shape
        device = data.device

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        tensor_attributes: Dict[str, TDCompatible],
        meta_attributes: Dict[str, Any],
    ) -> "TensorRelaxedBernoulli":
        """Reconstruct distribution from tensor attributes."""
        return cls(
            temperature=tensor_attributes["_temperature"],  # type: ignore
            probs=tensor_attributes.get("_probs"),  # type: ignore
            logits=tensor_attributes.get("_logits"),  # type: ignore
        )

    def dist(self) -> TorchRelaxedBernoulli:
        return TorchRelaxedBernoulli(
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

