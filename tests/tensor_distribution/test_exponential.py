"""
Tests for the TensorExponential distribution.

This module contains tests for the TensorExponential distribution, which wraps
`torch.distributions.Exponential`. The tests cover:
- Initialization with valid and invalid parameters.
- Correctness of distribution properties (mean, variance).
- `sample` and `rsample` methods.
- `log_prob` and `entropy` calculations.
- Behavior with different `reinterpreted_batch_ndims`.
"""

import pytest
import torch
from torch.testing import assert_close

from tensorcontainer.tensor_distribution.exponential import TensorExponential


class TestTensorExponentialInitialization:
    """
    Tests the initialization logic of the TensorExponential distribution.

    This suite verifies that:
    - The distribution can be created with a valid `rate` parameter.
    - Initialization fails when the `rate` parameter is not positive.
    """

    def test_valid_initialization(self):
        """The distribution should be created with a valid rate."""
        rate = torch.tensor([0.5, 1.0, 2.0])
        dist = TensorExponential(rate=rate, shape=rate.shape, device=rate.device)
        assert isinstance(dist, TensorExponential)
        assert_close(dist.rate, rate)

    @pytest.mark.parametrize(
        "rate",
        [
            torch.tensor([-0.1, 1.0]),  # Negative rate
            torch.tensor([0.0, 1.0]),  # Zero rate
        ],
    )
    def test_invalid_rate_raises_error(self, rate):
        """A ValueError should be raised for a non-positive rate."""
        with pytest.raises(ValueError):
            TensorExponential(rate=rate, shape=rate.shape, device=rate.device)


class TestTensorExponentialMethods:
    """
    Tests the methods of the TensorExponential distribution.

    This suite verifies that:
    - `sample` and `rsample` produce tensors of the correct shape and type.
    - `log_prob` computes the correct log probability.
    - `mean` and `variance` match the expected values.
    - `entropy` is calculated correctly.
    - The `dist` property returns the correct underlying torch distribution.
    """

    @pytest.fixture
    def dist(self):
        """Provides a standard TensorExponential distribution for testing."""
        rate = torch.tensor([0.5, 1.0, 4.0])
        return TensorExponential(rate=rate, shape=rate.shape, device=rate.device)

    def test_sample_shape(self, dist):
        """The shape of the sampled tensor should be correct."""
        sample = dist.sample()
        assert sample.shape == dist.shape

        samples = dist.sample(sample_shape=torch.Size([4, 4]))
        assert samples.shape == (4, 4) + dist.shape

    def test_rsample_shape(self, dist):
        """The shape of the r-sampled tensor should be correct and require grad."""
        dist.rate.requires_grad = True
        rsample = dist.rsample()
        assert rsample.shape == dist.shape
        assert rsample.requires_grad

    def test_log_prob(self, dist):
        """The log_prob should be consistent with the underlying torch distribution."""
        value = torch.tensor([1.0, 2.0, 0.5])
        expected_log_prob = dist.dist().log_prob(value)
        assert_close(dist.log_prob(value), expected_log_prob)

    def test_log_prob_for_negative_value(self, dist):
        """The log_prob should raise ValueError for negative values with validation enabled."""
        # Create a distribution without reinterpreted_batch_ndims for this test
        rate = torch.tensor([0.5, 1.0, 4.0])
        test_dist = TensorExponential(
            rate=rate, shape=rate.shape, device=rate.device, reinterpreted_batch_ndims=0
        )
        value = torch.tensor([-1.0, 2.0, 0.5])

        # With validation enabled, negative values should raise an error
        with pytest.raises(
            ValueError, match="Expected value argument.*to be within the support"
        ):
            test_dist.log_prob(value)

    def test_mean(self, dist):
        """The mean should match the formula 1 / rate."""
        expected_mean = 1 / dist.rate
        assert_close(dist.mean, expected_mean)

    def test_variance(self, dist):
        """The variance should match the formula 1 / rate^2."""
        expected_variance = 1 / (dist.rate**2)
        assert_close(dist.variance, expected_variance)

    def test_entropy(self, dist):
        """The entropy should be consistent with the underlying torch distribution."""
        expected_entropy = dist.dist().entropy()
        assert_close(dist.entropy(), expected_entropy)

    @pytest.mark.parametrize(
        "rbn_dims, expected_shape",
        [
            (0, (2, 3)),
            (1, (2,)),
            (2, ()),
        ],
    )
    def test_reinterpreted_batch_ndims(self, rbn_dims, expected_shape):
        """Tests log_prob with different reinterpreted_batch_ndims."""
        rate = torch.ones(2, 3)
        dist = TensorExponential(
            rate=rate,
            reinterpreted_batch_ndims=rbn_dims,
            shape=rate.shape,
            device=rate.device,
        )
        value = torch.rand(2, 3)
        log_prob = dist.log_prob(value)
        assert log_prob.shape == expected_shape
