from __future__ import annotations

from typing import Any, Dict, cast

from torch import Tensor
from torch.distributions import VonMises as TorchVonMises

from tensorcontainer.tensor_annotated import TDCompatible

from .base import TensorDistribution


class TensorVonMises(TensorDistribution):
    """Tensor-aware VonMises distribution."""

    # Annotated tensor parameters
    _loc: Tensor
    _concentration: Tensor

    def __init__(self, loc: Tensor, concentration: Tensor):
        # Parameter validation
        if loc is None:
            raise RuntimeError("'loc' must be provided.")
        if concentration is None:
            raise RuntimeError("'concentration' must be provided.")
        
        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        self._loc = loc
        self._concentration = concentration
        
        shape = loc.shape
        device = loc.device

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        tensor_attributes: Dict[str, TDCompatible],
        meta_attributes: Dict[str, Any],
    ) -> TensorVonMises:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            loc=cast(Tensor, tensor_attributes["_loc"]),
            concentration=cast(Tensor, tensor_attributes["_concentration"]),
        )

    def dist(self) -> TorchVonMises:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return TorchVonMises(
            loc=self._loc,
            concentration=self._concentration,
        )

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def loc(self) -> Tensor:
        """Returns the loc parameter of the distribution."""
        return self.dist().loc

    @property
    def concentration(self) -> Tensor:
        """Returns the concentration parameter of the distribution."""
        return self.dist().concentration
