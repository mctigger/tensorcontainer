from __future__ import annotations

from typing import Any, Dict

from torch import Tensor
from torch.distributions import Pareto as TorchPareto

from tensorcontainer.tensor_annotated import TDCompatible

from .base import TensorDistribution


class TensorPareto(TensorDistribution):
    """
    A Pareto distribution.

    This distribution is parameterized by `scale` and `alpha`.

    Source: https://pytorch.org/docs/stable/distributions.html#pareto
    """

    # Annotated tensor parameters
    _scale: Tensor
    _alpha: Tensor

    def __init__(self, scale: Tensor, alpha: Tensor) -> None:
        if scale is None:
            raise RuntimeError("`scale` must be provided.")
        if alpha is None:
            raise RuntimeError("`alpha` must be provided.")

        self._scale = scale
        self._alpha = alpha

        shape = scale.shape
        device = scale.device

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        tensor_attributes: Dict[str, TDCompatible],
        meta_attributes: Dict[str, Any],
    ) -> TensorPareto:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            scale=tensor_attributes["_scale"],  # type: ignore
            alpha=tensor_attributes["_alpha"],  # type: ignore
        )

    def dist(self) -> TorchPareto:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return TorchPareto(
            scale=self._scale,
            alpha=self._alpha,
        )

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def scale(self) -> Tensor:
        """Returns the scale parameter of the distribution."""
        return self.dist().scale

    @property
    def alpha(self) -> Tensor:
        """Returns the alpha parameter of the distribution."""
        return self.dist().alpha

