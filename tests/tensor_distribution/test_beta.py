"""
Tests for the TensorBeta distribution.

This module contains tests for the TensorBeta distribution, which wraps
`torch.distributions.Beta`. The tests cover:
- Initialization with valid and invalid parameters.
- Correctness of distribution properties (mean, variance).
- `sample` and `rsample` methods.
- `log_prob` and `entropy` calculations.
- Behavior with different `reinterpreted_batch_ndims`.
"""

import pytest
import torch
from torch.testing import assert_close

from tensorcontainer.tensor_distribution.beta import TensorBeta


class TestTensorBetaInitialization:
    """
    Tests the initialization logic of the TensorBeta distribution.

    This suite verifies that:
    - The distribution can be created with valid `concentration1` and `concentration0`.
    - Initialization fails when parameters have mismatching shapes.
    - Initialization fails when parameters are not positive.
    """

    def test_valid_initialization(self):
        """The distribution should be created with valid parameters."""
        concentration1 = torch.tensor([0.5, 2.0])
        concentration0 = torch.tensor([0.5, 3.0])
        dist = TensorBeta(
            concentration1=concentration1,
            concentration0=concentration0,
            shape=concentration1.shape,
            device=concentration1.device,
        )
        assert isinstance(dist, TensorBeta)
        assert_close(dist.concentration1, concentration1)
        assert_close(dist.concentration0, concentration0)

    @pytest.mark.parametrize(
        "concentration1, concentration0",
        [
            (torch.tensor([1.0, 2.0]), torch.tensor([1.0])),  # Mismatching shapes
            (torch.tensor([1.0]), torch.tensor([1.0, 2.0])),
        ],
    )
    def test_shape_mismatch_raises_error(self, concentration1, concentration0):
        """A ValueError should be raised for mismatching parameter shapes."""
        with pytest.raises(RuntimeError):
            TensorBeta(
                concentration1=concentration1,
                concentration0=concentration0,
                shape=concentration1.shape,
                device=concentration1.device,
            )

    @pytest.mark.parametrize(
        "concentration1, concentration0",
        [
            (torch.tensor([-0.1, 1.0]), torch.tensor([1.0, 1.0])),  # Negative value
            (torch.tensor([1.0, 1.0]), torch.tensor([0.0, 1.0])),  # Zero value
        ],
    )
    def test_invalid_parameter_values_raises_error(
        self, concentration1, concentration0
    ):
        """A ValueError should be raised for non-positive concentrations."""
        with pytest.raises(ValueError):
            TensorBeta(
                concentration1=concentration1,
                concentration0=concentration0,
                shape=concentration1.shape,
                device=concentration1.device,
            )


class TestTensorBetaMethods:
    """
    Tests the methods of the TensorBeta distribution.

    This suite verifies that:
    - `sample` and `rsample` produce tensors of the correct shape and type.
    - `log_prob` computes the correct log probability.
    - `mean` and `variance` match the expected values.
    - `entropy` is calculated correctly.
    - The `dist` property returns the correct underlying torch distribution.
    """

    @pytest.fixture
    def dist(self):
        """Provides a standard TensorBeta distribution for testing."""
        concentration1 = torch.tensor([0.5, 2.0, 5.0])
        concentration0 = torch.tensor([0.5, 3.0, 1.0])
        return TensorBeta(
            concentration1=concentration1,
            concentration0=concentration0,
            shape=concentration1.shape,
            device=concentration1.device,
        )

    def test_sample_shape(self, dist):
        """The shape of the sampled tensor should be correct."""
        sample = dist.sample()
        assert sample.shape == dist.shape

        samples = dist.sample(sample_shape=torch.Size([4, 4]))
        assert samples.shape == (4, 4) + dist.shape

    def test_rsample_shape(self, dist):
        """The shape of the r-sampled tensor should be correct and require grad."""
        dist.concentration1.requires_grad = True
        dist.concentration0.requires_grad = True
        rsample = dist.rsample()
        assert rsample.shape == dist.shape
        assert rsample.requires_grad

    def test_log_prob(self, dist):
        """The log_prob should be consistent with the underlying torch distribution."""
        value = torch.tensor([0.1, 0.5, 0.9])
        expected_log_prob = dist.dist().log_prob(value)
        assert_close(dist.log_prob(value), expected_log_prob)

    def test_mean(self, dist):
        """The mean should match the formula a / (a + b)."""
        expected_mean = dist.concentration1 / (
            dist.concentration1 + dist.concentration0
        )
        assert_close(dist.mean, expected_mean)

    def test_variance(self, dist):
        """The variance should match the formula ab / ((a+b)^2 * (a+b+1))."""
        a = dist.concentration1
        b = dist.concentration0
        expected_variance = (a * b) / ((a + b) ** 2 * (a + b + 1))
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
        concentration1 = torch.ones(2, 3) * 0.5
        concentration0 = torch.ones(2, 3) * 0.5
        dist = TensorBeta(
            concentration1=concentration1,
            concentration0=concentration0,
            reinterpreted_batch_ndims=rbn_dims,
            shape=concentration1.shape,
            device=concentration1.device,
        )
        value = torch.rand(2, 3)
        log_prob = dist.log_prob(value)
        assert log_prob.shape == expected_shape
