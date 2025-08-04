from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.distributions import Pareto as TorchPareto
from .utils import broadcast_all

from .base import TensorDistribution


class TensorPareto(TensorDistribution):
    """Tensor-aware Pareto distribution.

    Creates a Pareto distribution parameterized by `scale` and `alpha`.

    Args:
        scale: Scale parameter of the distribution. Must be positive.
        alpha: Shape parameter of the distribution. Must be positive.
        validate_args: Whether to validate the arguments of the distribution.

    Note:
        The Pareto distribution is defined for values greater than or equal to the scale parameter.
        It is commonly used to model phenomena with heavy tails, such as wealth distribution,
        city sizes, and natural phenomena.
    """

    # Annotated tensor parameters
    _scale: Tensor
    _alpha: Tensor

    def __init__(
        self,
        scale: float | Tensor,
        alpha: float | Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self._scale, self._alpha = broadcast_all(scale, alpha)

        super().__init__(self._scale.shape, self._scale.device, validate_args)

    @classmethod
    def _unflatten_distribution(cls, attributes: dict[str, Any]) -> TensorPareto:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            scale=attributes["_scale"],  # type: ignore
            alpha=attributes["_alpha"],  # type: ignore
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> TorchPareto:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return TorchPareto(
            scale=self._scale,
            alpha=self._alpha,
            validate_args=self._validate_args,
        )

    @property
    def scale(self) -> Tensor:
        """Returns the scale parameter of the distribution."""
        return self.dist().scale

    @property
    def alpha(self) -> Tensor:
        """Returns the alpha parameter of the distribution."""
        return self.dist().alpha
