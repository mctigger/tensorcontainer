from __future__ import annotations

import torch
from torch.distributions import (
    Normal,
    Transform,
    TransformedDistribution,
    constraints,
)
from typing import Any


def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Applies the symlog function element-wise.

    symlog(x) = sign(x) * log(1 + |x|)
    """
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """
    Applies the symexp function element-wise.

    symexp(x) = sign(x) * (exp(|x|) - 1)
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class SymexpTransform(Transform):
    """
    A bijective transform implementing the symexp function.

    This transform is its own inverse, applying symlog. It is used to warp a
    base distribution into a symlog-space.
    """

    def __init__(self) -> None:
        super().__init__()
        self.bijective = True
        self.domain = constraints.real
        self.codomain = constraints.real

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        return symexp(x)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        return symlog(y)

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # For y = symexp(x), dy/dx = exp(|x|)
        # log|dy/dx| = log(exp(|x|)) = |x|
        return torch.abs(x)

    @property
    def sign(self) -> int:
        """The sign of the transform (always positive for symexp)."""
        return 1


class SymLogDistribution(TransformedDistribution):
    """
    A distribution that transforms a Normal distribution with a symexp transform.

    This distribution is useful for modeling data with a wide dynamic range,
    where the data can be both positive and negative, and can have values
    close to zero. The symlog transform compresses large values and expands
    small values, making the distribution more stable for optimization.

    Args:
        loc (torch.Tensor): The mean of the base Normal distribution.
        scale (torch.Tensor): The standard deviation of the base Normal distribution.
        validate_args (bool, optional): Whether to validate the arguments.
            Defaults to None.
    """

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self._loc = loc
        self._scale = scale
        base_dist = Normal(loc, scale)
        super().__init__(base_dist, SymexpTransform(), validate_args=validate_args)

    @property
    def loc(self) -> torch.Tensor:
        return self._loc

    @property
    def scale(self) -> torch.Tensor:
        return self._scale

    @property
    def mean(self) -> torch.Tensor:
        """Approximated by mode for now, as per instructions."""
        return self.mode

    @property
    def mode(self) -> torch.Tensor:
        """The mode of the distribution."""
        return symexp(self._loc)

    def expand(
        self, batch_shape: Any, _instance: SymLogDistribution | None = None
    ) -> SymLogDistribution:
        """
        Returns a new distribution instance with expanded batch shape.

        Args:
            batch_shape (Any): The new batch shape.
            _instance (SymLogDistribution, optional): The instance to expand.
                Defaults to None.

        Returns:
            SymLogDistribution: The expanded distribution.
        """
        new = self._get_checked_instance(SymLogDistribution, _instance)
        batch_shape = torch.Size(batch_shape)
        new._loc = self._loc.expand(batch_shape)
        new._scale = self._scale.expand(batch_shape)
        base_dist = Normal(new._loc, new._scale)
        super(SymLogDistribution, new).__init__(
            base_dist, SymexpTransform(), validate_args=False
        )
        new._validate_args = self._validate_args
        return new
