"""
Tests for the TensorGamma distribution.

This module contains tests for the TensorGamma distribution, which wraps
`torch.distributions.Gamma`. The tests cover:
- Initialization with valid and invalid parameters.
- Correctness of distribution properties (mean, variance).
- `sample` and `rsample` methods.
- `log_prob` and `entropy` calculations.
- Behavior with different `reinterpreted_batch_ndims`.
"""

import pytest
import torch
from torch.testing import assert_close

from tensorcontainer.tensor_distribution.gamma import TensorGamma


class TestTensorGammaInitialization:
    """
    Tests the initialization logic of the TensorGamma distribution.

    This suite verifies that:
    - The distribution can be created with valid `concentration` and `rate`.
    - Initialization fails when parameters have mismatching shapes.
    - Initialization fails when parameters are not positive.
    """

    def test_valid_initialization(self):
        """The distribution should be created with valid parameters."""
        concentration = torch.tensor([0.5, 2.0])
        rate = torch.tensor([0.5, 3.0])
        dist = TensorGamma(
            concentration=concentration,
            rate=rate,
            shape=concentration.shape,
            device=concentration.device,
        )
        assert isinstance(dist, TensorGamma)
        assert_close(dist.concentration, concentration)
        assert_close(dist.rate, rate)

    @pytest.mark.parametrize(
        "concentration, rate",
        [
            (torch.tensor([1.0, 2.0]), torch.tensor([1.0])),  # Mismatching shapes
            (torch.tensor([1.0]), torch.tensor([1.0, 2.0])),
        ],
    )
    def test_shape_mismatch_raises_error(self, concentration, rate):
        """A ValueError should be raised for mismatching parameter shapes."""
        with pytest.raises(RuntimeError):
            TensorGamma(
                concentration=concentration,
                rate=rate,
                shape=concentration.shape,
                device=concentration.device,
            )

    @pytest.mark.parametrize(
        "concentration, rate",
        [
            (torch.tensor([-0.1, 1.0]), torch.tensor([1.0, 1.0])),  # Negative value
            (torch.tensor([1.0, 1.0]), torch.tensor([0.0, 1.0])),  # Zero value
        ],
    )
    def test_invalid_parameter_values_raises_error(self, concentration, rate):
        """A ValueError should be raised for non-positive concentration or rate."""
        with pytest.raises(ValueError):
            TensorGamma(
                concentration=concentration,
                rate=rate,
                shape=concentration.shape,
                device=concentration.device,
            )


class TestTensorGammaMethods:
    """
    Tests the methods of the TensorGamma distribution.

    This suite verifies that:
    - `sample` and `rsample` produce tensors of the correct shape and type.
    - `log_prob` computes the correct log probability.
    - `mean` and `variance` match the expected values.
    - `entropy` is calculated correctly.
    - The `dist` property returns the correct underlying torch distribution.
    """

    @pytest.fixture
    def dist(self):
        """Provides a standard TensorGamma distribution for testing."""
        concentration = torch.tensor([0.5, 2.0, 5.0])
        rate = torch.tensor([0.5, 3.0, 1.0])
        return TensorGamma(
            concentration=concentration,
            rate=rate,
            shape=concentration.shape,
            device=concentration.device,
        )

    def test_sample_shape(self, dist):
        """The shape of the sampled tensor should be correct."""
        sample = dist.sample()
        assert sample.shape == dist.shape

        samples = dist.sample(sample_shape=torch.Size([4, 4]))
        assert samples.shape == (4, 4) + dist.shape

    def test_rsample_shape(self, dist):
        """The shape of the r-sampled tensor should be correct and require grad."""
        dist.concentration.requires_grad = True
        dist.rate.requires_grad = True
        rsample = dist.rsample()
        assert rsample.shape == dist.shape
        assert rsample.requires_grad

    def test_log_prob(self, dist):
        """The log_prob should be consistent with the underlying torch distribution."""
        value = torch.tensor([1.0, 0.5, 5.0])
        expected_log_prob = dist.dist().log_prob(value)
        assert_close(dist.log_prob(value), expected_log_prob)

    def test_mean(self, dist):
        """The mean should match the formula concentration / rate."""
        expected_mean = dist.concentration / dist.rate
        assert_close(dist.mean, expected_mean)

    def test_variance(self, dist):
        """The variance should match the formula concentration / rate^2."""
        expected_variance = dist.concentration / (dist.rate**2)
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
        concentration = torch.ones(2, 3)
        rate = torch.ones(2, 3)
        dist = TensorGamma(
            concentration=concentration,
            rate=rate,
            reinterpreted_batch_ndims=rbn_dims,
            shape=concentration.shape,
            device=concentration.device,
        )
        value = torch.rand(2, 3)
        log_prob = dist.log_prob(value)
        assert log_prob.shape == expected_shape
