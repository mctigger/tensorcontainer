from __future__ import annotations

from typing import Any, Dict, Optional

from torch import Tensor
from torch.distributions import Gumbel as TorchGumbel

from tensorcontainer.tensor_annotated import TDCompatible

from .base import TensorDistribution


class TensorGumbel(TensorDistribution):
    """
    A Gumbel distribution.

    This distribution is parameterized by `loc` and `scale`.

    Source: https://pytorch.org/docs/stable/distributions.html#gumbel
    """

    _loc: Optional[Tensor] = None
    _scale: Optional[Tensor] = None

    def __init__(self, loc: Tensor, scale: Tensor):
        # Parameter validation occurs in super().__init__(), but we need an early
        # check here to safely derive shape and device from the data tensor
        # before calling the parent constructor
        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        self._loc = loc
        self._scale = scale

        if loc is None and scale is None:
            raise RuntimeError("Either 'loc' or 'scale' must be provided.")
        elif loc is not None:
            shape = loc.shape
            device = loc.device
        elif scale is not None:
            shape = scale.shape
            device = scale.device
        else:
            # This case should not be reached due to the check above
            raise RuntimeError("Unexpected error: No valid parameters provided.")

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        tensor_attributes: Dict[str, TDCompatible],
        meta_attributes: Dict[str, Any],
    ) -> TensorGumbel:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            loc=tensor_attributes.get("_loc"),  # type: ignore
            scale=tensor_attributes.get("_scale"),  # type: ignore
        )

    def dist(self) -> TorchGumbel:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return TorchGumbel(
            loc=self._loc,
            scale=self._scale,
        )

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def loc(self) -> Optional[Tensor]:
        """Returns the loc used to initialize the distribution."""
        return self.dist().loc

    @property
    def scale(self) -> Optional[Tensor]:
        """Returns the scale used to initialize the distribution."""
        return self.dist().scale

