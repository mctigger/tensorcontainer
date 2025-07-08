"""
Tests for the TensorBinomial distribution.

This module contains tests for the TensorBinomial distribution, which wraps
`torch.distributions.Binomial`. The tests cover:
- Initialization with `probs` and `logits`.
- Parameter validation for `total_count`, `probs`, and `logits`.
- Correctness of distribution properties (mean, variance).
- The `sample` method.
- `log_prob` and `entropy` calculations.
"""

import pytest
import torch
from torch.testing import assert_close

from tensorcontainer.tensor_distribution.binomial import TensorBinomial


class TestTensorBinomialInitialization:
    """
    Tests the initialization logic of the TensorBinomial distribution.

    This suite verifies that:
    - The distribution can be created with `total_count` and `probs`.
    - The distribution can be created with `total_count` and `logits`.
    - Initialization fails if both `probs` and `logits` are provided.
    - Initialization fails if `total_count` is not a non-negative integer.
    - Initialization fails if `probs` are not in the range [0, 1].
    """

    def test_valid_initialization_with_probs(self):
        """The distribution should be created with total_count and probs."""
        total_count = torch.tensor([10, 20])
        probs = torch.tensor([0.1, 0.8])
        dist = TensorBinomial(
            total_count=total_count,
            probs=probs,
            shape=probs.shape,
            device=probs.device,
        )
        assert isinstance(dist, TensorBinomial)
        assert_close(dist.total_count, total_count.float())
        assert_close(dist.probs, probs)

    def test_valid_initialization_with_logits(self):
        """The distribution should be created with total_count and logits."""
        total_count = torch.tensor([10, 20])
        logits = torch.tensor([-2.0, 1.5])
        dist = TensorBinomial(
            total_count=total_count,
            logits=logits,
            shape=logits.shape,
            device=logits.device,
        )
        assert isinstance(dist, TensorBinomial)
        assert_close(dist.total_count, total_count.float())
        assert_close(dist.logits, logits)

    def test_both_probs_and_logits_raises_error(self):
        """A ValueError should be raised if both probs and logits are given."""
        with pytest.raises(ValueError):
            total_count = torch.tensor(10)
            probs = torch.tensor(0.5)
            logits = torch.tensor(0.0)
            TensorBinomial(
                total_count=total_count,
                probs=probs,
                logits=logits,
                shape=total_count.shape,
                device=total_count.device,
            )

    @pytest.mark.parametrize("total_count", [-1, 10.5])
    def test_invalid_total_count_raises_error(self, total_count):
        """A ValueError should be raised for invalid total_count."""
        with pytest.raises(ValueError):
            total_count = torch.tensor(total_count)
            probs = torch.tensor(0.5)
            TensorBinomial(
                total_count=total_count,
                probs=probs,
                shape=total_count.shape,
                device=total_count.device,
            )

    @pytest.mark.parametrize("probs", [-0.1, 1.1])
    def test_invalid_probs_raises_error(self, probs):
        """A ValueError should be raised for probs outside [0, 1]."""
        with pytest.raises(ValueError):
            total_count = torch.tensor(10)
            probs = torch.tensor(probs)
            TensorBinomial(
                total_count=total_count,
                probs=probs,
                shape=total_count.shape,
                device=total_count.device,
            )


class TestTensorBinomialMethods:
    """
    Tests the methods of the TensorBinomial distribution.

    This suite verifies that:
    - `sample` produces tensors of the correct shape and type.
    - `log_prob` computes the correct log probability.
    - `mean` and `variance` match the expected values.
    - `entropy` is calculated correctly.
    """

    @pytest.fixture
    def dist(self):
        """Provides a standard TensorBinomial distribution for testing."""
        total_count = torch.tensor([10, 20, 30])
        probs = torch.tensor([0.1, 0.5, 0.9])
        return TensorBinomial(
            total_count=total_count,
            probs=probs,
            shape=probs.shape,
            device=probs.device,
        )

    def test_sample_shape(self, dist):
        """The shape of the sampled tensor should be correct."""
        sample = dist.sample()
        assert sample.shape == dist.shape

        samples = dist.sample(sample_shape=torch.Size([4, 4]))
        assert samples.shape == (4, 4) + dist.shape

    def test_log_prob(self, dist):
        """The log_prob should be consistent with the underlying torch distribution."""
        value = torch.tensor([1.0, 10.0, 27.0])
        expected_log_prob = dist.dist().log_prob(value)
        assert_close(dist.log_prob(value), expected_log_prob)

    def test_mean(self, dist):
        """The mean should match the formula total_count * probs."""
        expected_mean = dist.total_count * dist.probs
        assert_close(dist.mean, expected_mean)

    def test_variance(self, dist):
        """The variance should match the formula total_count * p * (1-p)."""
        expected_variance = dist.total_count * dist.probs * (1 - dist.probs)
        assert_close(dist.variance, expected_variance)

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_entropy(self, dist):
        """The entropy should be consistent with the underlying torch distribution."""
        expected_entropy = dist.dist().entropy()
        assert_close(dist.entropy(), expected_entropy)
