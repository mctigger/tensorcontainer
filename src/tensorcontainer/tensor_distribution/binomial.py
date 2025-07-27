from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
from torch import Size, Tensor
from torch.distributions import Binomial
from torch.distributions.utils import broadcast_all

from .base import TensorDistribution


class TensorBinomial(TensorDistribution):
    """Tensor-aware Binomial distribution.

    Creates a Binomial distribution parameterized by `total_count` and either `probs`
    or `logits` (but not both). The distribution represents the number of successes
    in `total_count` independent Bernoulli trials.

    Args:
        total_count: Number of Bernoulli trials. Can be an int or Tensor.
        probs: Event probabilities. Must be in range [0, 1]. Mutually exclusive with logits.
        logits: Event log-odds (log(p/(1-p))). Mutually exclusive with probs.
    """

    # Annotated tensor parameters
    _total_count: Tensor
    _probs: Optional[Tensor] = None
    _logits: Optional[Tensor] = None

    def __init__(
        self,
        total_count: Union[int, Tensor] = 1,
        probs: Optional[Tensor] = None,
        logits: Optional[Tensor] = None,
        validate_args: Optional[bool] = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )

        if probs is not None:
            self._total_count, self._probs = broadcast_all(total_count, probs)
        else:
            self._total_count, self._logits = broadcast_all(total_count, logits)

        shape = self._total_count.shape
        device = self._total_count.device

        super().__init__(shape, device, validate_args)

    @classmethod
    def _unflatten_distribution(cls, attributes: Dict[str, Any]):
        total_count = attributes["_total_count"]
        probs = attributes["_probs"]
        logits = attributes["_logits"]

        return cls(
            total_count=total_count,
            probs=probs,
            logits=logits,
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> Binomial:
        return Binomial(
            total_count=self._total_count,
            probs=self._probs,
            logits=self._logits,
            validate_args=self._validate_args,
        )

    @property
    def total_count(self) -> Tensor:
        """Returns the total_count parameter of the distribution."""
        return self._total_count

    @property
    def probs(self) -> Optional[Tensor]:
        """Returns the probs parameter of the distribution."""
        return self.dist().probs

    @property
    def logits(self) -> Optional[Tensor]:
        """Returns the logits parameter of the distribution."""
        return self.dist().logits

    @property
    def param_shape(self) -> Size:
        """Returns the shape of the underlying parameter."""
        return self.dist().param_shape
