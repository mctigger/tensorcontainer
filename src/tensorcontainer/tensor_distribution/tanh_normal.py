from __future__ import annotations

import math
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
from torch.nn import functional as F
from .utils import broadcast_all

from ..distributions.sampling import SamplingDistribution
from .base import TensorDistribution


class ClampedTanhTransform(torch.distributions.transforms.Transform):
    """Tanh transform whose inverse and log-determinant stay finite at saturation.

    ``torch.tanh`` rounds to exactly ``±1`` once ``|x|`` exceeds about 9 in float32,
    which a Normal with a scale of a few units samples routinely. Recovering ``x``
    from such a ``y`` with a bare ``atanh`` gives ``±inf``, and every density built on
    it -- ``log_prob``, hence the Monte-Carlo ``entropy`` and argmax ``mode`` of
    :class:`~tensorcontainer.distributions.sampling.SamplingDistribution` -- turns
    non-finite, with NaN gradients. Two guards keep the transform well-defined on the
    closed interval; both follow the ``TanhBijector`` of DreamerV2:

    * ``_inverse`` clamps ``y`` to the largest magnitude strictly below 1 that its
      dtype can represent before ``atanh``, so ``x`` is finite (about 8.7 for float32).
    * ``log_abs_det_jacobian`` uses the softplus identity
      ``log(1 - tanh(x)^2) = 2 (log 2 - x - softplus(-2x))``, evaluated on ``x``, which
      is exact and well-conditioned for every finite ``x``. The previous
      ``log(1 - y^2 + eps)`` was capped at ``log eps`` for saturated ``y`` and its
      derivative in ``y`` grows like ``1 / eps`` there.
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
        # The largest value below 1 that the dtype represents is 1 - eps / 2 (eps is
        # the spacing just above 1). Clamping there keeps atanh finite and makes the
        # gradient through a saturated sample zero instead of infinite.
        bound = 1.0 - torch.finfo(y.dtype).eps / 2
        return torch.atanh(y.clamp(-bound, bound))

    def log_abs_det_jacobian(self, x, y):
        # log|dy/dx| = log(1 - tanh(x)^2) = 2 * (log 2 - x - softplus(-2x)).
        # Written in x rather than y: y = tanh(x) is exactly ±1 in floating point
        # long before x is large, so 1 - y^2 underflows to 0 while this form does not.
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


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
