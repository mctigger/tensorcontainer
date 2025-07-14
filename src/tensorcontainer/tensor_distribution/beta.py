from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.distributions import Beta

from tensorcontainer.tensor_annotated import TDCompatible

from .base import TensorDistribution


class TensorBeta(TensorDistribution):
    """Tensor-aware Beta distribution."""
    
    # Annotated tensor parameters
    _concentration1: Optional[Tensor] = None
    _concentration0: Optional[Tensor] = None

    def __init__(self, concentration1: Optional[Tensor] = None, concentration0: Optional[Tensor] = None):
        # Parameter validation occurs in super().__init__(), but we need an early
        # check here to safely derive shape and device from the data tensor
        # before calling the parent constructor
        if concentration1 is None:
            raise RuntimeError("'concentration1' must be provided.")
        if concentration0 is None:
            raise RuntimeError("'concentration0' must be provided.")

        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        # Parameter validation occurs in super().__init__(), but we need an early
        # check here to safely derive shape and device from the data tensor
        # before calling the parent constructor
        if concentration1 is None and concentration0 is None:
            raise RuntimeError("Either 'concentration1' or 'concentration0' must be provided.")

        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        self._concentration1 = concentration1
        self._concentration0 = concentration0

        # Determine batch_shape and device
        if concentration1 is not None and concentration0 is not None:
            batch_shape = torch.broadcast_shapes(concentration1.shape, concentration0.shape)
            device = concentration1.device if concentration1.is_cuda else concentration0.device if concentration0.is_cuda else concentration1.device
        elif concentration1 is not None:
            batch_shape = concentration1.shape
            device = concentration1.device
        elif concentration0 is not None:
            batch_shape = concentration0.shape
            device = concentration0.device
        else:
            # This case should be caught by the RuntimeError above, but for type checking
            # and completeness, we'll assign default values.
            batch_shape = torch.Size([])
            device = torch.device("cpu")

        super().__init__(batch_shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        tensor_attributes: Dict[str, TDCompatible],
        meta_attributes: Dict[str, Any],
    ) -> TensorBeta:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            concentration1=tensor_attributes.get("_concentration1"),  # type: ignore
            concentration0=tensor_attributes.get("_concentration0"),  # type: ignore
        )

    def dist(self) -> Beta:
        return Beta(concentration1=self._concentration1, concentration0=self._concentration0)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def concentration1(self) -> Tensor:
        """Returns the concentration1 parameter of the distribution."""
        return self.dist().concentration1

    @property
    def concentration0(self) -> Tensor:
        """Returns the concentration0 parameter of the distribution."""
        return self.dist().concentration0

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the Beta distribution."""
        return self.dist().variance

    @property
    def mean(self) -> Tensor:
        """Returns the mean of the Beta distribution."""
        return self.dist().mean

