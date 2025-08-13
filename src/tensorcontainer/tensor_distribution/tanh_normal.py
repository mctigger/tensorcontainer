from __future__ import annotations

from functools import cached_property
from typing import Any

import torch
from torch import Tensor
from torch.distributions import (
    Distribution,
    Normal,
    TransformedDistribution,
    constraints,
)
from .utils import broadcast_all

from ..distributions.sampling import SamplingDistribution
from .base import TensorDistribution


class ClampedTanhTransform(torch.distributions.transforms.Transform):
    """
    Transform that applies tanh and clamps the output between -1 and 1.
    """

    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True

    @property
    def sign(self):
        return +1

    def __init__(self):
        super().__init__()

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        # Arctanh
        return torch.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # |det J| = 1 - tanh^2(x)
        # log|det J| = log(1 - tanh^2(x))
        # Use y = tanh(x) instead of recomputing tanh(x) for numerical stability
        return torch.log(
            1 - y.pow(2) + 1e-6
        )  # Adding small epsilon for numerical stability


class TensorTanhNormal(TensorDistribution):
    """Tensor-aware TanhNormal distribution.

    Creates a transformed Normal distribution where the output is passed through
    a hyperbolic tangent (tanh) function, constraining values to the interval (-1, 1).

    Args:
        loc: Location parameter of the underlying normal distribution.
        scale: Scale parameter of the underlying normal distribution. Must be positive.

    Note:
        This distribution is commonly used in reinforcement learning for bounded
        continuous action spaces. Use TensorIndependent to reinterpret batch dimensions
        as event dimensions if needed.
    """

    _loc: Tensor
    _scale: Tensor

    def __init__(
        self,
        loc: Tensor,
        scale: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self._loc, self._scale = broadcast_all(loc, scale)

        shape = self._loc.shape
        device = self._loc.device

        super().__init__(shape, device, validate_args)

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: dict[str, Any],
    ) -> TensorTanhNormal:
        return cls(
            loc=attributes["_loc"],
            scale=attributes["_scale"],
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> Distribution:
        return SamplingDistribution(
            TransformedDistribution(
                Normal(
                    self._loc.float(),
                    self._scale.float(),
                    validate_args=self._validate_args,
                ),
                [ClampedTanhTransform()],
                validate_args=self._validate_args,
            )
        )

    @property
    def loc(self) -> Tensor:
        """Returns the location parameter of the underlying normal distribution."""
        return self._loc

    @property
    def scale(self) -> Tensor:
        """Returns the scale parameter of the underlying normal distribution."""
        return self._scale

    @cached_property
    def _sampling_dist(self) -> SamplingDistribution:
        """Cached sampling distribution for consistent property calculations."""
        return SamplingDistribution(
            TransformedDistribution(
                Normal(
                    self._loc.float(),
                    self._scale.float(),
                    validate_args=self._validate_args,
                ),
                [
                    ClampedTanhTransform(),
                ],
                validate_args=self._validate_args,
            )
        )

    @property
    def mean(self) -> Tensor:
        """Returns the mean of the distribution."""
        return self._sampling_dist.mean

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the distribution."""
        return self._sampling_dist.variance

    @property
    def stddev(self) -> Tensor:
        """Returns the standard deviation of the distribution."""
        return self._sampling_dist.stddev
