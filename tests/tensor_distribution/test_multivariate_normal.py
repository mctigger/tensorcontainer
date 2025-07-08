"""
Tests for the TensorMultivariateNormal distribution.

This module contains tests for the TensorMultivariateNormal distribution, which
wraps `torch.distributions.MultivariateNormal`. The tests cover:
- Initialization with `loc` and `covariance_matrix`.
- Parameter validation for shapes and covariance matrix properties.
- Correctness of distribution properties (mean, variance).
- `sample` and `rsample` methods.
- `log_prob` and `entropy` calculations.
"""

import pytest
import torch
from torch.testing import assert_close

from tensorcontainer.tensor_distribution.multivariate_normal import (
    TensorMultivariateNormal,
)


class TestTensorMultivariateNormalInitialization:
    """
    Tests the initialization logic of the TensorMultivariateNormal distribution.

    This suite verifies that:
    - The distribution can be created with valid `loc` and `covariance_matrix`.
    - Initialization fails for shape mismatches between `loc` and `covariance_matrix`.
    - Initialization fails if the covariance matrix is not positive definite.
    """

    def test_valid_initialization(self):
        """The distribution should be created with valid parameters."""
        loc = torch.randn(3)
        cov = torch.eye(3)
        dist = TensorMultivariateNormal(
            loc=loc, covariance_matrix=cov, shape=loc.shape, device=loc.device
        )
        assert isinstance(dist, TensorMultivariateNormal)
        assert_close(dist.loc, loc)
        assert_close(dist.covariance_matrix, cov)

    def test_batched_initialization(self):
        """The distribution should support batched parameters."""
        loc = torch.randn(4, 3)
        cov = torch.eye(3).expand(4, 3, 3)
        dist = TensorMultivariateNormal(
            loc=loc, covariance_matrix=cov, shape=loc.shape, device=loc.device
        )
        assert dist.batch_shape == torch.Size([4])
        assert dist.event_shape == torch.Size([3])

    def test_shape_mismatch_raises_error(self):
        """A ValueError should be raised for mismatching parameter shapes."""
        loc = torch.randn(3)
        cov = torch.eye(4)  # Mismatching event shape
        with pytest.raises(RuntimeError):
            TensorMultivariateNormal(
                loc=loc, covariance_matrix=cov, shape=loc.shape, device=loc.device
            )

    def test_invalid_covariance_matrix_raises_error(self):
        """A ValueError should be raised for a non-positive-definite covariance matrix."""
        loc = torch.randn(3)
        cov = torch.ones(3, 3)  # Not positive definite
        with pytest.raises(ValueError):
            TensorMultivariateNormal(
                loc=loc, covariance_matrix=cov, shape=loc.shape, device=loc.device
            )


class TestTensorMultivariateNormalMethods:
    """
    Tests the methods of the TensorMultivariateNormal distribution.

    This suite verifies that:
    - `sample` and `rsample` produce tensors of the correct shape.
    - `log_prob` computes the correct log probability.
    - `mean`, `variance`, and `entropy` match expected values.
    """

    @pytest.fixture
    def dist(self):
        """Provides a standard TensorMultivariateNormal distribution."""
        loc = torch.tensor([1.0, -1.0])
        cov = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
        return TensorMultivariateNormal(
            loc=loc, covariance_matrix=cov, shape=loc.shape, device=loc.device
        )

    def test_sample_shape(self, dist):
        """The shape of the sampled tensor should be correct."""
        sample = dist.sample()
        assert sample.shape == dist.shape

        samples = dist.sample(sample_shape=torch.Size([4, 4]))
        assert samples.shape == (4, 4) + dist.shape

    def test_rsample_shape(self, dist):
        """The shape of the r-sampled tensor should be correct and require grad."""
        dist.loc.requires_grad = True
        # covariance_matrix does not support gradients for rsample
        rsample = dist.rsample()
        assert rsample.shape == dist.shape
        assert rsample.requires_grad

    def test_log_prob(self, dist):
        """The log_prob should be consistent with the underlying torch distribution."""
        value = torch.tensor([1.1, -0.9])
        expected_log_prob = dist.dist().log_prob(value)
        assert_close(dist.log_prob(value), expected_log_prob)

    def test_mean(self, dist):
        """The mean should match the loc."""
        assert_close(dist.mean, dist.loc)

    def test_variance(self, dist):
        """The variance should be the diagonal of the covariance matrix."""
        expected_variance = torch.diag(dist.covariance_matrix)
        assert_close(dist.variance, expected_variance)

    def test_entropy(self, dist):
        """The entropy should be consistent with the underlying torch distribution."""
        expected_entropy = dist.dist().entropy()
        assert_close(dist.entropy(), expected_entropy)
