import torch
from torch.distributions import Distribution, constraints
from functools import cached_property
from typing import Any, Optional


class SamplingDistribution(Distribution):
    """
    A wrapper for a PyTorch distribution that calculates statistics via sampling.

    This distribution is useful when the analytical statistics of a base
    distribution are not available or not desired. Instead, it computes
    properties like mean, stddev, variance, and mode by drawing samples from the
    base distribution.

    To improve efficiency, it caches the generated samples and the computed
    statistics, ensuring that repeated access to these properties does not
    trigger redundant computations.

    Args:
        base_distribution (Distribution): The underlying distribution to sample from.
        n (int, optional): The number of samples to draw for calculating
            statistics. Defaults to 100.
    """

    __slots__ = ["base_dist", "n"]

    def __init__(self, base_distribution: Distribution, *, n: int = 100):
        if not isinstance(base_distribution, Distribution):
            raise TypeError(
                "base_distribution must be a torch.distributions.Distribution"
            )
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")

        self.base_dist = base_distribution
        self.n = n

        super().__init__(
            batch_shape=self.base_dist.batch_shape,
            event_shape=self.base_dist.event_shape,
            validate_args=False,  # We defer validation to the base distribution
        )

    def __repr__(self) -> str:
        return f"SamplingDistribution(base_dist={self.base_dist}, n={self.n})"

    def __getattr__(self, name: str) -> Any:
        """Delegates attribute access to the base distribution."""
        return getattr(self.base_dist, name)

    @cached_property
    def _samples(self) -> torch.Tensor:
        """
        Cached samples from the base distribution.

        Uses rsample if available for reparameterization-friendly gradients,
        otherwise falls back to sample.
        """
        sample_shape = torch.Size((self.n,))
        if self.base_dist.has_rsample:
            return self.base_dist.rsample(sample_shape)
        return self.base_dist.sample(sample_shape)

    @property
    def has_rsample(self) -> bool:
        return self.base_dist.has_rsample

    def rsample(self, sample_shape: Any = torch.Size()) -> torch.Tensor:
        """Delegates rsample to the base distribution."""
        return self.base_dist.rsample(sample_shape)

    def sample(self, sample_shape: Any = torch.Size()) -> torch.Tensor:
        """Delegates sample to the base distribution."""
        return self.base_dist.sample(sample_shape)

    @cached_property
    def mean(self) -> torch.Tensor:  # type: ignore
        """Mean of the distribution, computed as the mean of cached samples."""
        return self._samples.float().mean(0)

    @cached_property
    def stddev(self) -> torch.Tensor:  # type: ignore
        """Standard deviation of the distribution, computed from cached samples."""
        return self._samples.float().std(0)

    @cached_property
    def variance(self) -> torch.Tensor:  # type: ignore
        """Variance of the distribution, computed from cached samples."""
        return self._samples.float().var(0)

    @cached_property
    def mode(self) -> torch.Tensor:  # type: ignore
        """
        Mode of the distribution.

        Tries to return the analytical mode if available. Otherwise, it computes
        the mode via Monte Carlo approximation by finding the sample with the
        highest log probability.
        """
        try:
            return self.base_dist.mode
        except (AttributeError, NotImplementedError):
            pass  # Fall back to sampling

        log_probs = self.base_dist.log_prob(self._samples)
        max_indices = torch.argmax(log_probs, dim=0)

        # Use advanced indexing to gather the modes efficiently
        return self._samples.gather(
            0, max_indices.reshape(1, *max_indices.shape, *(1,) * len(self.event_shape))
        ).squeeze(0)

    def entropy(self) -> torch.Tensor:
        """
        Entropy of the distribution, estimated via Monte Carlo.

        Calculates the negative mean of the log probabilities of cached samples.
        """
        log_prob = self.base_dist.log_prob(self._samples)
        return -log_prob.mean(0)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Delegates log probability calculation to the base distribution."""
        return self.base_dist.log_prob(value)

    @property
    def support(self) -> Optional[constraints.Constraint]:
        """Delegates support to the base distribution."""
        return self.base_dist.support

    @property
    def arg_constraints(self) -> dict:
        """Delegates argument constraints to the base distribution."""
        return self.base_dist.arg_constraints
