"""
Tests for the TensorPoisson distribution.

This module contains tests for the TensorPoisson distribution, which wraps
`torch.distributions.Poisson`. The tests cover:
- Initialization with a valid `rate`.
- Parameter validation for `rate`.
- Correctness of distribution properties (mean, variance).
- The `sample` method.
- `log_prob` and `entropy` calculations.
"""

import pytest
import torch
from torch.testing import assert_close

from tensorcontainer.tensor_distribution.poisson import TensorPoisson


class TestTensorPoissonInitialization:
    """
    Tests the initialization logic of the TensorPoisson distribution.

    This suite verifies that:
    - The distribution can be created with a valid `rate` parameter.
    - Initialization fails when the `rate` parameter is not positive.
    """

    def test_valid_initialization(self):
        """The distribution should be created with a valid rate."""
        rate = torch.tensor([0.5, 1.0, 10.0])
        dist = TensorPoisson(rate=rate, shape=rate.shape, device=rate.device)
        assert isinstance(dist, TensorPoisson)
        assert_close(dist.rate, rate)

    @pytest.mark.parametrize(
        "rate",
        [
            torch.tensor([-0.1, 1.0]),  # Negative rate
        ],
    )
    def test_invalid_rate_raises_error(self, rate):
        """A ValueError should be raised for a negative rate."""
        with pytest.raises(ValueError):
            TensorPoisson(rate=rate, shape=rate.shape, device=rate.device)

    def test_zero_rate_is_valid(self):
        """Zero rate should be valid for Poisson distribution."""
        rate = torch.tensor([0.0, 1.0])
        dist = TensorPoisson(rate=rate, shape=rate.shape, device=rate.device)
        assert isinstance(dist, TensorPoisson)
        assert_close(dist.rate, rate)


class TestTensorPoissonMethods:
    """
    Tests the methods of the TensorPoisson distribution.

    This suite verifies that:
    - `sample` produces tensors of the correct shape and type.
    - `log_prob` computes the correct log probability.
    - `mean` and `variance` match the expected values.
    - `entropy` is calculated correctly.
    """

    @pytest.fixture
    def dist(self):
        """Provides a standard TensorPoisson distribution for testing."""
        rate = torch.tensor([0.5, 2.0, 10.0])
        return TensorPoisson(rate=rate, shape=rate.shape, device=rate.device)

    def test_sample_shape(self, dist):
        """The shape of the sampled tensor should be correct."""
        sample = dist.sample()
        assert sample.shape == dist.shape

        samples = dist.sample(sample_shape=torch.Size([4, 4]))
        assert samples.shape == (4, 4) + dist.shape

    def test_log_prob(self, dist):
        """The log_prob should be consistent with the underlying torch distribution."""
        value = torch.tensor([0.0, 2.0, 10.0])
        expected_log_prob = dist.dist().log_prob(value)
        assert_close(dist.log_prob(value), expected_log_prob)

    def test_mean(self, dist):
        """The mean should match the rate."""
        expected_mean = dist.rate
        assert_close(dist.mean, expected_mean)

    def test_variance(self, dist):
        """The variance should match the rate."""
        expected_variance = dist.rate
        assert_close(dist.variance, expected_variance)

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_entropy(self, dist):
        """The entropy should be consistent with the underlying torch distribution."""
        # Poisson entropy does not have a simple closed-form expression,
        # so we check consistency with the torch implementation.
        expected_entropy = dist.dist().entropy()
        assert_close(dist.entropy(), expected_entropy)
