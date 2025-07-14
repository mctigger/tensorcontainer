from __future__ import annotations

from typing import Any, Dict, Optional

from torch import Tensor
from torch.distributions import Exponential

from tensorcontainer.tensor_annotated import TDCompatible

from .base import TensorDistribution


class TensorExponential(TensorDistribution):
    """Tensor-aware Exponential distribution."""
    
    # Annotated tensor parameters
    _rate: Optional[Tensor] = None

    def __init__(self, rate: Tensor):
        # Parameter validation occurs in super().__init__(), but we need an early
        # check here to safely derive shape and device from the data tensor
        # before calling the parent constructor
        if rate is None:
            raise RuntimeError("'rate' must be provided.")
        
        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        self._rate = rate
        
        shape = rate.shape
        device = rate.device

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        tensor_attributes: Dict[str, TDCompatible],
        meta_attributes: Dict[str, Any],
    ) -> TensorExponential:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            rate=tensor_attributes.get("_rate"),  # type: ignore
        )

    def dist(self) -> Exponential:
        return Exponential(rate=self._rate)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def rate(self) -> Tensor:
        """Returns the rate parameter used to initialize the distribution."""
        return self.dist().rate

    @property
    def mean(self) -> Tensor:
        """Returns the mean of the distribution."""
        return self.dist().mean

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the distribution."""
        return self.dist().variance

    @property
    def support(self) -> Any:
        """Returns the support of the distribution."""
        return self.dist().support

    @property
    def arg_constraints(self) -> Dict[str, Any]:
        """Returns the argument constraints of the distribution."""
        return self.dist().arg_constraints

    @property
    def has_rsample(self) -> bool:
        """Returns True if the distribution has a reparameterization trick."""
        return self.dist().has_rsample

    @property
    def has_enumerate_support(self) -> bool:
        """Returns True if the distribution has enumerate_support."""
        return self.dist().has_enumerate_support
