
from __future__ import annotations

from typing import Any, Dict, Optional

from torch import Size, Tensor
from torch.distributions import ContinuousBernoulli as TorchContinuousBernoulli

from tensorcontainer.tensor_annotated import TDCompatible

from .base import TensorDistribution


class ContinuousBernoulli(TensorDistribution):
    """Tensor-aware Continuous Bernoulli distribution."""

    # Annotated tensor parameters
    _probs: Optional[Tensor] = None
    _logits: Optional[Tensor] = None
    _lims: tuple[float, float] = (0.499, 0.501)

    def __init__(
        self,
        probs: Optional[Tensor] = None,
        logits: Optional[Tensor] = None,
        lims: tuple[float, float] = (0.499, 0.501),
    ) -> None:
        data = probs if probs is not None else logits
        if data is None:
            raise RuntimeError("Either 'probs' or 'logits' must be provided.")

        # Store the parameters in annotated attributes before calling super().__init__()
        self._probs = probs
        self._logits = logits
        self._lims = lims

        shape = data.shape
        device = data.device

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        tensor_attributes: Dict[str, TDCompatible],
        meta_attributes: Dict[str, Any],
    ) -> ContinuousBernoulli:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            probs=tensor_attributes.get("_probs"),  # type: ignore
            logits=tensor_attributes.get("_logits"),  # type: ignore
            lims=meta_attributes.get("_lims", (0.499, 0.501)),
        )

    def dist(self) -> TorchContinuousBernoulli:
        return TorchContinuousBernoulli(
            probs=self._probs, logits=self._logits, lims=self._lims
        )

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

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
