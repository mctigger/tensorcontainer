"""
Tests for the TensorLogNormal distribution.

This module contains tests for the TensorLogNormal distribution, which wraps
`torch.distributions.LogNormal`. The tests cover:
- Initialization with valid and invalid parameters.
- Correctness of distribution properties (mean, variance).
- `sample` and `rsample` methods.
- `log_prob` and `entropy` calculations.
- Behavior with different `reinterpreted_batch_ndims`.
"""

import pytest
import torch
from torch.testing import assert_close

from tensorcontainer.tensor_distribution.log_normal import TensorLogNormal


class TestTensorLogNormalInitialization:
    """
    Tests the initialization logic of the TensorLogNormal distribution.

    This suite verifies that:
    - The distribution can be created with valid `loc` and `scale`.
    - Initialization fails when parameters have mismatching shapes.
    - Initialization fails when scale is not positive.
    """

    def test_valid_initialization(self):
        """The distribution should be created with valid parameters."""
        loc = torch.tensor([0.0, 2.0])
        scale = torch.tensor([1.0, 3.0])
        dist = TensorLogNormal(
            loc=loc, scale=scale, shape=torch.Size([2]), device=torch.device("cpu")
        )
        assert isinstance(dist, TensorLogNormal)
        assert_close(dist.loc, loc)
        assert_close(dist.scale, scale)

    @pytest.mark.parametrize(
        "loc, scale",
        [
            (torch.tensor([1.0, 2.0]), torch.tensor([1.0])),  # Mismatching shapes
            (torch.tensor([1.0]), torch.tensor([1.0, 2.0])),
        ],
    )
    def test_shape_mismatch_raises_error(self, loc, scale):
        """A ValueError should be raised for mismatching parameter shapes."""
        with pytest.raises(RuntimeError):
            TensorLogNormal(
                loc=loc,
                scale=scale,
                shape=loc.shape,
                device=loc.device,
            )

    @pytest.mark.parametrize(
        "loc, scale",
        [
            (torch.tensor([0.0, 1.0]), torch.tensor([-1.0, 1.0])),  # Negative value
            (torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])),  # Zero value
        ],
    )
    def test_invalid_parameter_values_raises_error(self, loc, scale):
        """A ValueError should be raised for non-positive scale."""
        with pytest.raises(ValueError):
            TensorLogNormal(
                loc=loc, scale=scale, shape=torch.Size([2]), device=torch.device("cpu")
            )


class TestTensorLogNormalMethods:
    """
    Tests the methods of the TensorLogNormal distribution.

    This suite verifies that:
    - `sample` and `rsample` produce tensors of the correct shape and type.
    - `log_prob` computes the correct log probability.
    - `mean` and `variance` match the expected values.
    - `entropy` is calculated correctly.
    - The `dist` property returns the correct underlying torch distribution.
    """

    @pytest.fixture
    def dist(self):
        """Provides a standard TensorLogNormal distribution for testing."""
        return TensorLogNormal(
            loc=torch.tensor([0.0, 2.0, 5.0]),
            scale=torch.tensor([1.0, 3.0, 2.0]),
            shape=torch.Size([3]),
            device=torch.device("cpu"),
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
        dist.scale.requires_grad = True
        rsample = dist.rsample()
        assert rsample.shape == dist.shape
        assert rsample.requires_grad

    def test_log_prob(self, dist):
        """The log_prob should be consistent with the underlying torch distribution."""
        value = torch.tensor([1.0, 0.5, 5.0])
        expected_log_prob = dist.dist().log_prob(value)
        assert_close(dist.log_prob(value), expected_log_prob)

    def test_mean(self, dist):
        """The mean should match the formula exp(loc + scale^2 / 2)."""
        expected_mean = torch.exp(dist.loc + dist.scale.pow(2) / 2)
        assert_close(dist.mean, expected_mean)

    def test_variance(self, dist):
        """The variance should be consistent with the underlying torch distribution."""
        expected_variance = dist.dist().variance
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
        loc = torch.ones(2, 3)
        scale = torch.ones(2, 3)
        dist = TensorLogNormal(
            loc=loc,
            scale=scale,
            reinterpreted_batch_ndims=rbn_dims,
            shape=torch.Size([2, 3]),
            device=torch.device("cpu"),
        )
        value = torch.rand(2, 3)
        log_prob = dist.log_prob(value)
        assert log_prob.shape == expected_shape
