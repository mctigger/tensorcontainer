"""
Tests for the TensorGeometric distribution.

This module contains tests for the TensorGeometric distribution, which wraps
`torch.distributions.Geometric`. The tests cover:
- Initialization with `probs` and `logits`.
- Parameter validation for `probs` and `logits`.
- Correctness of distribution properties (mean, variance).
- The `sample` method.
- `log_prob` and `entropy` calculations.
"""

import pytest
import torch
from torch.testing import assert_close

from tensorcontainer.tensor_distribution.geometric import TensorGeometric


class TestTensorGeometricInitialization:
    """
    Tests the initialization logic of the TensorGeometric distribution.

    This suite verifies that:
    - The distribution can be created with `probs`.
    - The distribution can be created with `logits`.
    - Initialization fails if both `probs` and `logits` are provided.
    - Initialization fails if `probs` are not in the range [0, 1].
    """

    def test_valid_initialization_with_probs(self):
        """The distribution should be created with probs."""
        probs = torch.tensor([0.1, 0.5, 0.9])
        dist = TensorGeometric(probs=probs, shape=probs.shape, device=probs.device)
        assert isinstance(dist, TensorGeometric)
        assert_close(dist.probs, probs)

    def test_valid_initialization_with_logits(self):
        """The distribution should be created with logits."""
        logits = torch.tensor([-2.0, 0.0, 2.0])
        dist = TensorGeometric(logits=logits, shape=logits.shape, device=logits.device)
        assert isinstance(dist, TensorGeometric)
        assert_close(dist.logits, logits)

    def test_both_probs_and_logits_raises_error(self):
        """A ValueError should be raised if both probs and logits are given."""
        with pytest.raises(ValueError):
            probs = torch.tensor(0.5)
            logits = torch.tensor(0.0)
            TensorGeometric(
                probs=probs,
                logits=logits,
                shape=probs.shape,
                device=probs.device,
            )

    @pytest.mark.parametrize("probs", [-0.1, 1.1, 0.0, 1.0])
    def test_invalid_probs_raises_error(self, probs):
        """A ValueError should be raised for probs outside (0, 1)."""
        with pytest.raises(ValueError):
            probs = torch.tensor(probs)
            TensorGeometric(probs=probs, shape=probs.shape, device=probs.device)


class TestTensorGeometricMethods:
    """
    Tests the methods of the TensorGeometric distribution.

    This suite verifies that:
    - `sample` produces tensors of the correct shape and type.
    - `log_prob` computes the correct log probability.
    - `mean` and `variance` match the expected values.
    - `entropy` is calculated correctly.
    """

    @pytest.fixture
    def dist(self):
        """Provides a standard TensorGeometric distribution for testing."""
        probs = torch.tensor([0.1, 0.5, 0.9])
        return TensorGeometric(probs=probs, shape=probs.shape, device=probs.device)

    def test_sample_shape(self, dist):
        """The shape of the sampled tensor should be correct."""
        sample = dist.sample()
        assert sample.shape == dist.shape

        samples = dist.sample(sample_shape=torch.Size([4, 4]))
        assert samples.shape == (4, 4) + dist.shape

    def test_log_prob(self, dist):
        """The log_prob should be consistent with the underlying torch distribution."""
        value = torch.tensor([1.0, 2.0, 0.0])
        expected_log_prob = dist.dist().log_prob(value)
        assert_close(dist.log_prob(value), expected_log_prob)

    def test_mean(self, dist):
        """The mean should match the formula 1/p."""
        expected_mean = 1 / dist.probs
        assert_close(dist.mean, expected_mean)

    def test_variance(self, dist):
        """The variance should match the formula (1-p)/p^2."""
        expected_variance = (1 - dist.probs) / (dist.probs**2)
        assert_close(dist.variance, expected_variance)

    def test_entropy(self, dist):
        """The entropy should be consistent with the underlying torch distribution."""
        expected_entropy = dist.dist().entropy()
        assert_close(dist.entropy(), expected_entropy)
