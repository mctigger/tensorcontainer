from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Geometric
from torch.distributions.utils import broadcast_all
from torch.types import Number

from .base import TensorDistribution


class TensorGeometric(TensorDistribution):
    """Tensor-aware Geometric distribution."""

    # Annotated tensor parameters
    _probs: Optional[Tensor]
    _logits: Optional[Tensor]

    def __init__(
        self,
        probs: Optional[Union[Number, Tensor]] = None,
        logits: Optional[Union[Number, Tensor]] = None,
        validate_args: Optional[bool] = None,
    ):
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
    def _unflatten_distribution(
        cls,
        attributes: Dict[str, Any],
    ) -> TensorGeometric:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            probs=attributes.get("_probs"),
            logits=attributes.get("_logits"),
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> Geometric:
        return Geometric(
            probs=self._probs, logits=self._logits, validate_args=self._validate_args
        )

    @property
    def probs(self) -> Optional[Tensor]:
        """Returns the probabilities used to initialize the distribution."""
        return self.dist().probs

    @property
    def logits(self) -> Optional[Tensor]:
        """Returns the logits used to initialize the distribution."""
        return self.dist().logits
