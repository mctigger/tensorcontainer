"""
Tests for the TensorMultinomial distribution.

This module contains tests for the TensorMultinomial distribution, which wraps
`torch.distributions.Multinomial`. The tests cover:
- Initialization with valid and invalid parameters.
- Correctness of distribution properties (mean, variance).
- `sample` method.
- `log_prob` calculations.
- Behavior with different `reinterpreted_batch_ndims`.
"""

import pytest
import torch
from torch.testing import assert_close

from tensorcontainer.tensor_distribution.multinomial import TensorMultinomial


class TestTensorMultinomialInitialization:
    """
    Tests the initialization logic of the TensorMultinomial distribution.

    This suite verifies that:
    - The distribution can be created with valid `total_count` and `probs`/`logits`.
    - Initialization fails when `total_count` is not a positive integer.
    - Initialization fails when both or neither of `probs` and `logits` are provided.
    """

    def test_valid_initialization_with_probs(self):
        """The distribution should be created with valid `total_count` and `probs`."""
        probs = torch.tensor([0.1, 0.2, 0.7])
        dist = TensorMultinomial(
            total_count=10,
            probs=probs,
            shape=torch.Size([3]),
            device=torch.device("cpu"),
        )
        assert isinstance(dist, TensorMultinomial)
        assert dist.total_count == 10
        assert_close(dist.probs, probs)

    def test_valid_initialization_with_logits(self):
        """The distribution should be created with valid `total_count` and `logits`."""
        logits = torch.tensor([-1.0, 0.0, 1.0])
        dist = TensorMultinomial(
            total_count=10,
            logits=logits,
            shape=torch.Size([3]),
            device=torch.device("cpu"),
        )
        assert isinstance(dist, TensorMultinomial)
        assert dist.total_count == 10
        assert_close(dist.logits, logits)

    @pytest.mark.parametrize("total_count", [-1, 0])
    def test_invalid_total_count_raises_error(self, total_count):
        """A ValueError should be raised for non-positive total_count."""
        with pytest.raises(ValueError):
            TensorMultinomial(
                total_count=total_count,
                probs=torch.tensor([0.5, 0.5]),
                shape=torch.Size([2]),
                device=torch.device("cpu"),
            )

    def test_probs_and_logits_raises_error(self):
        """A ValueError should be raised if both probs and logits are provided."""
        with pytest.raises(ValueError):
            TensorMultinomial(
                total_count=10,
                probs=torch.tensor([0.5, 0.5]),
                logits=torch.tensor([-1.0, 1.0]),
                shape=torch.Size([2]),
                device=torch.device("cpu"),
            )

    def test_no_probs_or_logits_raises_error(self):
        """A ValueError should be raised if neither probs nor logits are provided."""
        with pytest.raises(ValueError):
            TensorMultinomial(
                total_count=10, shape=torch.Size([]), device=torch.device("cpu")
            )


class TestTensorMultinomialMethods:
    """
    Tests the methods of the TensorMultinomial distribution.

    This suite verifies that:
    - `sample` produces tensors of the correct shape and type.
    - `log_prob` computes the correct log probability.
    - `mean` and `variance` match the expected values.
    - The `dist` property returns the correct underlying torch distribution.
    """

    @pytest.fixture
    def dist_probs(self):
        """Provides a standard TensorMultinomial distribution for testing."""
        return TensorMultinomial(
            total_count=10,
            probs=torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
            shape=torch.Size([2, 2]),
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
        value = torch.tensor([[1.0, 9.0], [8.0, 2.0]])
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
        probs = torch.ones(2, 3, 4) / 4
        dist = TensorMultinomial(
            total_count=10,
            probs=probs,
            reinterpreted_batch_ndims=rbn_dims,
            shape=torch.Size([2, 3, 4]),
            device=torch.device("cpu"),
        )
        value = torch.randint(0, 10, (2, 3, 4)).float()
        value = value / value.sum(-1, True) * 10
        log_prob = dist.log_prob(value)
        assert log_prob.shape == expected_shape
