from __future__ import annotations

from typing import Any, Dict, Optional, Union

from torch import Size, Tensor
from torch.distributions import NegativeBinomial
from torch.distributions.utils import broadcast_all

from .base import TensorDistribution


class TensorNegativeBinomial(TensorDistribution):
    """Tensor-aware NegativeBinomial distribution."""

    # Annotated tensor parameters
    _total_count: Tensor
    _probs: Optional[Tensor] = None
    _logits: Optional[Tensor] = None

    def __init__(
        self,
        total_count: Union[float, Tensor],
        probs: Optional[Tensor] = None,
        logits: Optional[Tensor] = None,
        validate_args: Optional[bool] = None,
    ):
        if probs is not None and logits is not None:
            raise ValueError("Only one of 'probs' or 'logits' can be specified.")

        if probs is not None:
            self._total_count, self._probs = broadcast_all(total_count, probs)
            self._logits = None
        elif logits is not None:
            self._total_count, self._logits = broadcast_all(total_count, logits)
            self._probs = None
        else:
            raise ValueError("Either 'probs' or 'logits' must be provided.")

        shape = self._total_count.shape
        device = self._total_count.device

        super().__init__(shape, device, validate_args)

    @classmethod
    def _unflatten_distribution(
        cls, attributes: Dict[str, Any]
    ) -> TensorNegativeBinomial:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            total_count=attributes["_total_count"],
            probs=attributes.get("_probs"),
            logits=attributes.get("_logits"),
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> NegativeBinomial:
        return NegativeBinomial(
            total_count=self._total_count,
            probs=self._probs,
            logits=self._logits,
            validate_args=self._validate_args,
        )

    @property
    def total_count(self) -> Tensor:
        """Returns the total_count used to initialize the distribution."""
        return self.dist().total_count

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
