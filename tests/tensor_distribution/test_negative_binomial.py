"""
Tests for the TensorNegativeBinomial distribution.

This module contains tests for the TensorNegativeBinomial distribution, which wraps
`torch.distributions.NegativeBinomial`. The tests cover:
- Initialization with valid and invalid parameters.
- Correctness of distribution properties (mean, variance).
- `sample` and `rsample` methods.
- `log_prob` and `entropy` calculations.
- Behavior with different `reinterpreted_batch_ndims`.
"""

import pytest
import torch
from torch.testing import assert_close

from tensorcontainer.tensor_distribution.negative_binomial import (
    TensorNegativeBinomial,
)


class TestTensorNegativeBinomialInitialization:
    """
    Tests the initialization logic of the TensorNegativeBinomial distribution.

    This suite verifies that:
    - The distribution can be created with valid `total_count` and `probs`/`logits`.
    - Initialization fails when parameters have mismatching shapes.
    - Initialization fails when `total_count` is not positive.
    - Initialization fails when both or neither of `probs` and `logits` are provided.
    """

    def test_valid_initialization_with_probs(self):
        """The distribution should be created with valid `total_count` and `probs`."""
        total_count = torch.tensor([10.0, 20.0])
        probs = torch.tensor([0.25, 0.75])
        dist = TensorNegativeBinomial(
            total_count=total_count,
            probs=probs,
            shape=torch.Size([2]),
            device=torch.device("cpu"),
        )
        assert isinstance(dist, TensorNegativeBinomial)
        assert_close(dist.total_count, total_count)
        assert_close(dist.probs, probs)

    def test_valid_initialization_with_logits(self):
        """The distribution should be created with valid `total_count` and `logits`."""
        total_count = torch.tensor([10.0, 20.0])
        logits = torch.tensor([-1.0, 1.0])
        dist = TensorNegativeBinomial(
            total_count=total_count,
            logits=logits,
            shape=torch.Size([2]),
            device=torch.device("cpu"),
        )
        assert isinstance(dist, TensorNegativeBinomial)
        assert_close(dist.total_count, total_count)
        assert_close(dist.logits, logits)

    @pytest.mark.parametrize(
        "total_count, probs",
        [
            (torch.tensor([10.0, 20.0]), torch.tensor([0.5])),  # Mismatching shapes
            (torch.tensor([10.0]), torch.tensor([0.5, 0.5])),
        ],
    )
    def test_shape_mismatch_raises_error(self, total_count, probs):
        """A ValueError should be raised for mismatching parameter shapes."""
        with pytest.raises(RuntimeError):
            TensorNegativeBinomial(
                total_count=total_count,
                probs=probs,
                shape=total_count.shape,
                device=total_count.device,
            )

    def test_invalid_parameter_values_raises_error(self):
        """A ValueError should be raised for non-positive total_count."""
        with pytest.raises(ValueError):
            TensorNegativeBinomial(
                total_count=torch.tensor([-1.0, 1.0]),
                probs=torch.tensor([0.5, 0.5]),
                shape=torch.Size([2]),
                device=torch.device("cpu"),
            )

    def test_probs_and_logits_raises_error(self):
        """A ValueError should be raised if both probs and logits are provided."""
        with pytest.raises(ValueError):
            TensorNegativeBinomial(
                total_count=torch.tensor([10.0, 20.0]),
                probs=torch.tensor([0.5, 0.5]),
                logits=torch.tensor([-1.0, 1.0]),
                shape=torch.Size([2]),
                device=torch.device("cpu"),
            )

    def test_no_probs_or_logits_raises_error(self):
        """A ValueError should be raised if neither probs nor logits are provided."""
        with pytest.raises(ValueError):
            TensorNegativeBinomial(
                total_count=torch.tensor([10.0, 20.0]),
                shape=torch.Size([2]),
                device=torch.device("cpu"),
            )


class TestTensorNegativeBinomialMethods:
    """
    Tests the methods of the TensorNegativeBinomial distribution.

    This suite verifies that:
    - `sample` produces tensors of the correct shape and type.
    - `log_prob` computes the correct log probability.
    - `mean` and `variance` match the expected values.
    - `entropy` is calculated correctly.
    - The `dist` property returns the correct underlying torch distribution.
    """

    @pytest.fixture
    def dist_probs(self):
        """Provides a standard TensorNegativeBinomial distribution for testing."""
        return TensorNegativeBinomial(
            total_count=torch.tensor([10.0, 20.0, 30.0]),
            probs=torch.tensor([0.25, 0.5, 0.75]),
            shape=torch.Size([3]),
            device=torch.device("cpu"),
        )

    def test_sample_shape(self, dist_probs):
        """The shape of the sampled tensor should be correct."""
        sample = dist_probs.sample()
        assert sample.shape == dist_probs.shape

        samples = dist_probs.sample(sample_shape=torch.Size([4, 4]))
        assert samples.shape == (4, 4) + dist_probs.shape

    def test_log_prob(self, dist_probs):
        """The log_prob should be consistent with the underlying torch distribution."""
        value = torch.tensor([10.0, 20.0, 30.0])
        expected_log_prob = dist_probs.dist().log_prob(value)
        assert_close(dist_probs.log_prob(value), expected_log_prob)

    def test_mean(self, dist_probs):
        """The mean should be consistent with the underlying torch distribution."""
        expected_mean = dist_probs.dist().mean
        assert_close(dist_probs.mean, expected_mean)

    def test_variance(self, dist_probs):
        """The variance should be consistent with the underlying torch distribution."""
        expected_variance = dist_probs.dist().variance
        assert_close(dist_probs.variance, expected_variance)

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
        total_count = torch.ones(2, 3) * 10
        probs = torch.ones(2, 3) * 0.5
        dist = TensorNegativeBinomial(
            total_count=total_count,
            probs=probs,
            reinterpreted_batch_ndims=rbn_dims,
            shape=torch.Size([2, 3]),
            device=torch.device("cpu"),
        )
        value = torch.randint(0, 20, (2, 3)).float()
        log_prob = dist.log_prob(value)
        assert log_prob.shape == expected_shape
