from __future__ import annotations

from typing import Any, Dict

from torch import Size, Tensor
from torch.distributions import Dirichlet

from tensorcontainer.tensor_annotated import TDCompatible

from .base import TensorDistribution


class TensorDirichlet(TensorDistribution):
    """Tensor-aware Dirichlet distribution."""

    # Annotated tensor parameters
    _concentration: Tensor

    def __init__(self, concentration: Tensor):
        # Parameter validation occurs in super().__init__(), but we need an early
        # check here to safely derive shape and device from the data tensor
        # before calling the parent constructor
        if concentration is None:
            raise RuntimeError("`concentration` must be provided.")

        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        self._concentration = concentration

        shape = concentration.shape[:-1]
        device = concentration.device

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        tensor_attributes: Dict[str, TDCompatible],
        meta_attributes: Dict[str, Any],
    ) -> TensorDirichlet:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            concentration=tensor_attributes.get("_concentration"),  # type: ignore
        )

    def dist(self) -> Dirichlet:
        return Dirichlet(concentration=self._concentration)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the Dirichlet distribution."""
        # For Dirichlet, variance is a matrix, but we return the diagonal (marginal variances)
        alpha = self._concentration
        alpha_sum = alpha.sum(-1, keepdim=True)
        return (alpha * (alpha_sum - alpha)) / (alpha_sum**2 * (alpha_sum + 1))

    @property
    def param_shape(self) -> Size:
        """Returns the shape of the underlying parameter."""
        return self._concentration.shape
