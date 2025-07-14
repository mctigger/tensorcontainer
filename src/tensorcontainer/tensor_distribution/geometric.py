from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.distributions import Geometric

from tensorcontainer.tensor_annotated import TDCompatible

from .base import TensorDistribution


class TensorGeometric(TensorDistribution):
    """Tensor-aware Geometric distribution."""

    # Annotated tensor parameters
    _probs: Optional[Tensor] = None
    _logits: Optional[Tensor] = None

    def __init__(self, probs: Optional[Tensor] = None, logits: Optional[Tensor] = None):
        data = probs if probs is not None else logits
        # Parameter validation occurs in super().__init__(), but we need an early
        # check here to safely derive shape and device from the data tensor
        # before calling the parent constructor
        if data is None:
            raise RuntimeError("Either 'probs' or 'logits' must be provided.")
        if probs is not None and logits is not None:
            raise RuntimeError("Only one of 'probs' or 'logits' can be provided.")

        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
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
    ) -> TensorGeometric:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            probs=tensor_attributes.get("_probs"),  # type: ignore
            logits=tensor_attributes.get("_logits"),  # type: ignore
        )

    def dist(self) -> Geometric:
        return Geometric(probs=self._probs, logits=self._logits)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def probs(self) -> Optional[Tensor]:
        """Returns the probabilities used to initialize the distribution."""
        return self.dist().probs

    @property
    def logits(self) -> Optional[Tensor]:
        """Returns the logits used to initialize the distribution."""
        return self.dist().logits

    @property
    def mean(self) -> Tensor:
        """Returns the mean of the distribution."""
        return self.dist().mean

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the distribution."""
        return self.dist().variance

    @property
    def stddev(self) -> Tensor:
        """Returns the standard deviation of the distribution."""
        return self.dist().stddev

    def entropy(self) -> Tensor:
        """Returns the entropy of the distribution."""
        return self.dist().entropy()

    @property
    def support(self) -> Any:
        """Returns the support of the distribution."""
        return self.dist().support

    @property
    def arg_constraints(self) -> Dict[str, Any]:
        """Returns the argument constraints of the distribution."""
        return self.dist().arg_constraints

    @property
    def batch_shape(self) -> torch.Size:
        """Returns the batch shape of the distribution."""
        return self.dist().batch_shape

    @property
    def event_shape(self) -> torch.Size:
        """Returns the event shape of the distribution."""
        return self.dist().event_shape

    @property
    def has_rsample(self) -> bool:
        """Returns True if the distribution has a reparameterization trick."""
        return self.dist().has_rsample

    @property
    def has_enumerate_support(self) -> bool:
        """Returns True if the distribution has enumerate_support implemented."""
        return self.dist().has_enumerate_support

    @property
    def _validate_args(self) -> bool:
        """Returns True if the distribution validates arguments."""
        return self.dist()._validate_args

