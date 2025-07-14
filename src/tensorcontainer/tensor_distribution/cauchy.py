from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.distributions import Cauchy

from tensorcontainer.tensor_annotated import TDCompatible

from .base import TensorDistribution


class TensorCauchy(TensorDistribution):
    """Tensor-aware Cauchy distribution.
    
    Creates a Cauchy distribution parameterized by `loc` (location) and `scale` parameters.
    The Cauchy distribution is a continuous probability distribution with heavy tails.
    
    Args:
        loc: Location parameter (median) of the distribution.
        scale: Scale parameter of the distribution. Must be positive.
        
    Note:
        The Cauchy distribution has no finite mean or variance. These properties
        are not implemented as they would return undefined values.
    """
    
    # Annotated tensor parameters
    _loc: Optional[Tensor] = None
    _scale: Optional[Tensor] = None

    def __init__(self, loc: Tensor, scale: Tensor):
        # Parameter validation occurs in super().__init__(), but we need an early
        # check here to safely derive shape and device from the data tensor
        # before calling the parent constructor
        if loc is None or scale is None:
            raise RuntimeError("Both 'loc' and 'scale' must be provided.")
        
        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        shape = torch.broadcast_shapes(loc.shape, scale.shape)
        device = loc.device
        self._loc = loc.expand(shape)
        self._scale = scale.expand(shape)

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        tensor_attributes: Dict[str, TDCompatible],
        meta_attributes: Dict[str, Any],
    ) -> TensorCauchy:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            loc=tensor_attributes.get("_loc"),  # type: ignore
            scale=tensor_attributes.get("_scale"),  # type: ignore
        )

    def dist(self) -> Cauchy:
        return Cauchy(loc=self._loc, scale=self._scale)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def loc(self) -> Tensor:
        """Returns the location parameter of the distribution."""
        return self.dist().loc

    @property
    def scale(self) -> Tensor:
        """Returns the scale parameter of the distribution."""
        return self.dist().scale
