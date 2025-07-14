from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
from torch import Size, Tensor
from torch.distributions import Binomial

from tensorcontainer.tensor_annotated import TDCompatible

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
    ):
        # Parameter validation occurs in super().__init__(), but we need an early
        # check here to safely derive shape and device from the data tensor
        # before calling the parent constructor
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        
        # Derive shape and device from the non-None parameter
        data = probs if probs is not None else logits
        assert data is not None
        
        # Convert int total_count to tensor with proper broadcasting
        if isinstance(total_count, int):
            total_count = torch.full(data.shape, total_count, dtype=data.dtype, device=data.device)
        else:
            # Ensure total_count has the same dtype as the data parameter
            total_count = total_count.to(dtype=data.dtype, device=data.device)
        
        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        self._total_count = total_count
        self._probs = probs
        self._logits = logits
        
        shape = torch.broadcast_shapes(total_count.shape, data.shape)
        device = data.device

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        tensor_attributes: Dict[str, TDCompatible],
        meta_attributes: Dict[str, Any],
    ) -> TensorBinomial:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            total_count=tensor_attributes.get("_total_count"),  # type: ignore
            probs=tensor_attributes.get("_probs"),  # type: ignore
            logits=tensor_attributes.get("_logits"),  # type: ignore
        )

    def dist(self) -> Binomial:
        return Binomial(
            total_count=self._total_count, probs=self._probs, logits=self._logits  # type: ignore
        )

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def total_count(self) -> Tensor:
        """Returns the total_count parameter of the distribution."""
        return self.dist().total_count

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

    @property
    def mean(self) -> Tensor:
        """Returns the mean of the Binomial distribution."""
        return self.dist().mean

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the Binomial distribution."""
        return self.dist().variance
