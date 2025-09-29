from __future__ import annotations

import torch
from torch import Size, Tensor
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all


class DiracDistribution(Distribution):
    """
    Dirac delta distribution (point mass distribution).

    A degenerate discrete distribution that assigns probability one to the single
    element in its support. This distribution concentrates all probability mass
    at a specific value.

    Args:
        value: The single support element where all probability mass is concentrated.
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
    """

    arg_constraints = {"value": constraints.real}
    support = constraints.real
    has_rsample = True

    def __init__(self, value: Tensor, validate_args: bool | None = None):
        (self.value,) = broadcast_all(value)
        super().__init__(self.value.shape, validate_args=validate_args)

    def expand(self, batch_shape: Size, _instance=None) -> DiracDistribution:
        """Expand the distribution to a new batch shape."""
        new = self._get_checked_instance(DiracDistribution, _instance)
        batch_shape = torch.Size(batch_shape)
        new.value = self.value.expand(batch_shape)
        super(DiracDistribution, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape: Size = Size()) -> Tensor:
        """Generate reparameterized samples from the distribution."""
        shape = self._extended_shape(sample_shape)
        return self.value.expand(shape)

    def sample(self, sample_shape: Size = Size()) -> Tensor:
        """Generate samples from the distribution."""
        return self.rsample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        """
        Compute the log probability density of the given value.

        Returns 0.0 for values that exactly match the distribution's value,
        and -inf for all other values.
        """
        if self._validate_args:
            self._validate_sample(value)
        # Use torch.eq for exact equality check
        is_equal = torch.eq(value, self.value)
        return torch.where(is_equal, 0.0, -torch.inf)

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

        Returns 0 for values less than the point mass, 1 for values greater
        than or equal to the point mass.
        """
        if self._validate_args:
            self._validate_sample(value)
        return torch.where(value >= self.value, 1.0, 0.0)

    def icdf(self, value: Tensor) -> Tensor:
        """
        Inverse cumulative distribution function.

        Returns the point value for any probability > 0.
        """
        if self._validate_args and not torch.all((value >= 0) & (value <= 1)):
            raise ValueError("The value argument must be within [0, 1]")
        return torch.full_like(
            value, self.value.item() if self.value.numel() == 1 else float("nan")
        )
