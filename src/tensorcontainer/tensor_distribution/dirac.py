from __future__ import annotations

from typing import Any

from torch import Tensor
from tensorcontainer.distributions import DiracDistribution

from .base import TensorDistribution
from .utils import broadcast_all


class TensorDirac(TensorDistribution):
    """
    Tensor-aware Dirac delta distribution (point mass distribution).

    A degenerate discrete distribution that assigns probability one to the single
    element in its support. This distribution concentrates all probability mass
    at a specific value.

    Args:
        value: The single support element where all probability mass is concentrated.
        atol: Absolute tolerance for comparing closeness to value. Default is 0.
        rtol: Relative tolerance for comparing closeness to value. Default is 0.
        validate_args: Whether to validate distribution parameters.

    Example:
        >>> import torch
        >>> from tensorcontainer.tensor_distribution import TensorDirac
        >>> value = torch.tensor([1.0, 2.0, 3.0])
        >>> dist = TensorDirac(value)
        >>> dist.sample()
        tensor([1., 2., 3.])
        >>> dist.log_prob(value)
        tensor([0., 0., 0.])
        >>> dist.log_prob(torch.tensor([0.0, 2.0, 4.0]))
        tensor([-inf, 0., -inf])
        >>> # With tolerance
        >>> dist_tol = TensorDirac(torch.tensor([1.0]), atol=0.1)
        >>> dist_tol.log_prob(torch.tensor([1.05]))
        tensor([0.])
    """

    # Annotated tensor parameters
    _value: Tensor
    _atol: Tensor | None
    _rtol: Tensor | None

    def __init__(
        self,
        value: Tensor,
        atol: Tensor | float | None = 0.0,
        rtol: Tensor | float | None = 0.0,
        validate_args: bool | None = None,
    ):
        # Use broadcast_all to ensure value is a tensor
        (self._value,) = broadcast_all(value)
        self._atol = atol
        self._rtol = rtol
        super().__init__(self._value.shape, self._value.device, validate_args)

    @classmethod
    def _unflatten_distribution(cls, attributes: dict[str, Any]) -> TensorDirac:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            value=attributes["_value"],
            atol=attributes.get("_atol", 0.0),
            rtol=attributes.get("_rtol", 0.0),
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> DiracDistribution:
        """Return the underlying DiracDistribution instance."""
        return DiracDistribution(
            self._value,
            atol=self._atol,
            rtol=self._rtol,
            validate_args=self._validate_args,
        )

    @property
    def value(self) -> Tensor:
        """The point value where all probability mass is concentrated."""
        return self._value
