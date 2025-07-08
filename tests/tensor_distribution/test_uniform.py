"""
Tests for the TensorUniform distribution.

This module contains tests for the TensorUniform distribution, which wraps
`torch.distributions.Uniform`. The tests cover:
- Initialization with valid and invalid parameters.
- Correctness of distribution properties (mean, stddev).
- `sample` and `rsample` methods.
- `log_prob` and `entropy` calculations.
- Behavior with different `reinterpreted_batch_ndims`.
"""

import pytest
import torch
from torch.testing import assert_close

from tensorcontainer.tensor_distribution.uniform import TensorUniform


class TestTensorUniformInitialization:
    """
    Tests the initialization logic of the TensorUniform distribution.

    This suite verifies that:
    - The distribution can be created with valid `low` and `high` parameters.
    - Initialization fails when parameters have mismatching shapes.
    - Initialization fails when `low` is not strictly less than `high`.
    """

    def test_valid_initialization(self):
        """The distribution should be created with valid parameters."""
        low = torch.tensor([0.0, -1.0])
        high = torch.tensor([1.0, 1.0])
        dist = TensorUniform(low=low, high=high, shape=low.shape, device=low.device)
        assert isinstance(dist, TensorUniform)
        assert_close(dist.low, low)
        assert_close(dist.high, high)

    @pytest.mark.parametrize(
        "low, high",
        [
            (torch.tensor([0.0, 1.0]), torch.tensor([1.0])),  # Mismatching shapes
            (torch.tensor([0.0]), torch.tensor([1.0, 2.0])),
        ],
    )
    def test_shape_mismatch_raises_error(self, low, high):
        """A ValueError should be raised for mismatching parameter shapes."""
        with pytest.raises(RuntimeError):
            TensorUniform(low=low, high=high, shape=low.shape, device=low.device)

    @pytest.mark.parametrize(
        "low, high",
        [
            (torch.tensor([1.0, 1.0]), torch.tensor([0.0, 1.0])),  # low > high
            (torch.tensor([1.0, 1.0]), torch.tensor([1.0, 2.0])),  # low == high
        ],
    )
    def test_invalid_parameter_values_raises_error(self, low, high):
        """A ValueError should be raised if low >= high."""
        with pytest.raises(ValueError):
            TensorUniform(low=low, high=high, shape=low.shape, device=low.device)


class TestTensorUniformMethods:
    """
    Tests the methods of the TensorUniform distribution.

    This suite verifies that:
    - `sample` and `rsample` produce tensors of the correct shape and type.
    - `log_prob` computes the correct log probability.
    - `mean` and `stddev` match the expected values.
    - `entropy` is calculated correctly.
    - The `dist` property returns the correct underlying torch distribution.
    """

    @pytest.fixture
    def dist(self):
        """Provides a standard TensorUniform distribution for testing."""
        low = torch.tensor([-1.0, 0.0, 10.0])
        high = torch.tensor([1.0, 5.0, 11.0])
        return TensorUniform(low=low, high=high, shape=low.shape, device=low.device)

    def test_sample_shape(self, dist):
        """The shape of the sampled tensor should be correct."""
        sample = dist.sample()
        assert sample.shape == dist.shape

        samples = dist.sample(sample_shape=torch.Size([4, 4]))
        assert samples.shape == (4, 4) + dist.shape

    def test_rsample_shape(self, dist):
        """The shape of the r-sampled tensor should be correct and require grad."""
        dist.low.requires_grad = True
        dist.high.requires_grad = True
        rsample = dist.rsample()
        assert rsample.shape == dist.shape
        assert rsample.requires_grad

    def test_log_prob(self, dist):
        """The log_prob should be consistent with the underlying torch distribution."""
        value = torch.tensor([0.0, 2.5, 10.5])
        expected_log_prob = dist.dist().log_prob(value)
        assert_close(dist.log_prob(value), expected_log_prob)

    def test_log_prob_outside_support(self, dist):
        """The log_prob of a value outside the support should be -inf."""
        value = torch.tensor([-2.0, 6.0, 9.0])  # Outside support
        log_prob = dist.log_prob(value)
        assert torch.all(log_prob == -float("inf"))

    def test_mean(self, dist):
        """The mean should match the formula (low + high) / 2."""
        expected_mean = (dist.low + dist.high) / 2
        assert_close(dist.mean, expected_mean)

    def test_stddev(self, dist):
        """The standard deviation should be correct."""
        expected_variance = ((dist.high - dist.low) ** 2) / 12
        expected_stddev = torch.sqrt(expected_variance)
        assert_close(dist.stddev, expected_stddev)

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
        low = torch.zeros(2, 3)
        high = torch.ones(2, 3)
        dist = TensorUniform(
            low=low,
            high=high,
            reinterpreted_batch_ndims=rbn_dims,
            shape=low.shape,
            device=low.device,
        )
        value = torch.rand(2, 3)
        log_prob = dist.log_prob(value)
        assert log_prob.shape == expected_shape
