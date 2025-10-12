from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

from tensorcontainer.types import ShapeLike


class DiracDistribution(Distribution):
    """
    Dirac delta distribution (point mass distribution).

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
        >>> from tensorcontainer.distributions import DiracDistribution
        >>> value = torch.tensor([1.0, 2.0, 3.0])
        >>> dist = DiracDistribution(value)
        >>> dist.sample()
        tensor([1., 2., 3.])
        >>> dist.log_prob(value)
        tensor([0., 0., 0.])
        >>> dist.log_prob(torch.tensor([0.0, 2.0, 4.0]))
        tensor([-inf, 0., -inf])
        >>> # With tolerance
        >>> dist_tol = DiracDistribution(torch.tensor([1.0]), atol=0.1)
        >>> dist_tol.log_prob(torch.tensor([1.05]))
        tensor([0.])
    """

    arg_constraints = {
        "value": constraints.real,
        "atol": constraints.greater_than_eq(0),
        "rtol": constraints.greater_than_eq(0),
    }
    support = constraints.real
    has_rsample = True

    def __init__(
        self,
        value: Tensor,
        atol: Tensor | float | None = 0.0,
        rtol: Tensor | float | None = 0.0,
        validate_args: bool | None = None,
    ):
        # Broadcast all parameters
        self.value, self.atol, self.rtol = broadcast_all(value, atol, rtol)
        super().__init__(self.value.shape, validate_args=validate_args)

    def expand(self, batch_shape: ShapeLike, _instance=None) -> DiracDistribution:
        """Expand the distribution to a new batch shape."""
        new = self._get_checked_instance(DiracDistribution, _instance)
        batch_shape = torch.Size(batch_shape)
        new.value = self.value.expand(batch_shape)
        new.atol = self.atol.expand(batch_shape)
        new.rtol = self.rtol.expand(batch_shape)
        super(DiracDistribution, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape: ShapeLike = torch.Size()) -> Tensor:
        """Generate reparameterized samples from the distribution."""
        shape = self._extended_shape(sample_shape)
        return self.value.expand(shape)

    def sample(self, sample_shape: ShapeLike = torch.Size()) -> Tensor:
        """Generate samples from the distribution."""
        return self.rsample(sample_shape)

    def _slack(self, value: Tensor) -> Tensor:
        """
        Compute the tolerance slack for comparing values.

        The slack is computed as: atol + rtol * |value|
        """
        return self.atol + self.rtol * torch.abs(value)

    def log_prob(self, value: Tensor) -> Tensor:
        """
        Compute the log probability density of the given value.

        Returns 0.0 for values within tolerance of the distribution's value,
        and -inf for all other values. With default tolerances (atol=0, rtol=0),
        this uses exact equality.
        """
        if self._validate_args:
            self._validate_sample(value)
        # Check if value is within tolerance: |value - self.value| <= slack
        is_within_tolerance = torch.abs(value - self.value) <= self._slack(self.value)
        return torch.where(is_within_tolerance, 0.0, -torch.inf)

    @property
    def mean(self) -> Tensor:
        """Mean of the distribution (the point value)."""
        return self.value

    @property
    def mode(self) -> Tensor:
        """Mode of the distribution (the point value)."""
        return self.value

    @property
    def variance(self) -> Tensor:
        """Variance of the distribution (always zero for point mass)."""
        return torch.zeros_like(self.value)

    @property
    def stddev(self) -> Tensor:
        """Standard deviation of the distribution (always zero for point mass)."""
        return torch.zeros_like(self.value)

    def entropy(self) -> Tensor:
        """Entropy of the distribution (always zero for point mass)."""
        return torch.zeros_like(self.value)

    def cdf(self, value: Tensor) -> Tensor:
        """
        Cumulative distribution function.

        Returns 0 for values less than (point mass - slack), 1 for values greater
        than or equal to (point mass - slack). With default tolerances (atol=0, rtol=0),
        this uses exact comparison.
        """
        if self._validate_args:
            self._validate_sample(value)
        # CDF is 1 for values >= self.value - slack
        return torch.where(value >= self.value - self._slack(self.value), 1.0, 0.0)

    def icdf(self, value: Tensor) -> Tensor:
        """
        Inverse cumulative distribution function.

        Returns the point value for any probability > 0.
        """
        if self._validate_args and not torch.all((value >= 0) & (value <= 1)):
            raise ValueError("The value argument must be within [0, 1]")
        # For a Dirac delta, icdf returns the point value for any p in (0, 1]
        # We need to broadcast the value tensor to match the shape of the input
        return torch.broadcast_to(self.value, value.shape)
