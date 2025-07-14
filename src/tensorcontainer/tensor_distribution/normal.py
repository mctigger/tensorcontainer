from __future__ import annotations

from typing import Any, Dict, Optional, cast

import torch
from torch import Tensor
from torch.distributions import Normal

from tensorcontainer.tensor_annotated import TDCompatible

from .base import TensorDistribution


class TensorNormal(TensorDistribution):
    """Tensor-aware Normal distribution.

    Creates a Normal distribution parameterized by `loc` (mean) and `scale` (standard deviation).

    Args:
        loc: Mean of the distribution.
        scale: Standard deviation of the distribution. Must be positive.

    Note:
        The Normal distribution is also known as the Gaussian distribution.
    """

    # Annotated tensor parameters
    _loc: Optional[Tensor] = None
    _scale: Optional[Tensor] = None

    def __init__(self, loc: Tensor, scale: Tensor):
        # Parameter validation occurs in super().__init__(), but we need an early
        # check here to safely derive shape and device from the data tensor
        # before calling the parent constructor
        if loc is None:
            raise RuntimeError("'loc' must be provided.")
        if scale is None:
            raise RuntimeError("'scale' must be provided.")

        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        batch_shape = torch.broadcast_shapes(loc.shape, scale.shape)
        device = loc.device # Prioritize CUDA device if available
        self._loc = loc
        self._scale = scale

        super().__init__(batch_shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        tensor_attributes: Dict[str, TDCompatible],
        meta_attributes: Dict[str, Any],
    ) -> TensorNormal:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            loc=cast(Tensor, tensor_attributes.get("_loc")),
            scale=cast(Tensor, tensor_attributes.get("_scale")),
        )

    def dist(self) -> Normal:
        """Return Normal distribution."""
        return Normal(
            loc=self._loc,
            scale=self._scale,
        )

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
