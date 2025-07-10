"""
Tests for the TensorDirichlet distribution.

This module contains tests for the TensorDirichlet distribution, which wraps
`torch.distributions.Dirichlet`. The tests cover:
- Initialization with a valid `concentration` parameter.
- Parameter validation for the `concentration` vector.
- Correctness of distribution properties (mean).
- `sample` and `rsample` methods.
- `log_prob` and `entropy` calculations.
"""

import pytest
import torch
from torch.testing import assert_close

from tensorcontainer.tensor_distribution.dirichlet import TensorDirichlet


class TestTensorDirichletInitialization:
    """
    Tests the initialization logic of the TensorDirichlet distribution.

    This suite verifies that:
    - The distribution can be created with a valid `concentration` vector.
    - Initialization fails if the `concentration` contains non-positive values.
    """

    def test_valid_initialization(self):
        """The distribution should be created with a valid concentration vector."""
        concentration = torch.tensor([0.5, 1.0, 5.0])
        dist = TensorDirichlet(
            concentration=concentration,
            shape=concentration.shape,
            device=concentration.device,
        )
        assert isinstance(dist, TensorDirichlet)
        assert_close(dist.concentration, concentration)

    def test_batched_initialization(self):
        """The distribution should support batched parameters."""
        concentration = torch.rand(4, 3) + 0.1  # Ensure positive
        dist = TensorDirichlet(
            concentration=concentration,
            shape=concentration.shape,
            device=concentration.device,
        )
        assert dist.batch_shape == torch.Size([4])
        assert dist.event_shape == torch.Size([3])

    @pytest.mark.parametrize(
        "concentration",
        [
            torch.tensor([-0.1, 1.0, 2.0]),  # Negative value
            torch.tensor([0.0, 1.0, 2.0]),  # Zero value
        ],
    )
    def test_invalid_concentration_raises_error(self, concentration):
        """A ValueError should be raised for a non-positive concentration."""
        with pytest.raises(ValueError):
            TensorDirichlet(
                concentration=concentration,
                shape=concentration.shape,
                device=concentration.device,
            )


class TestTensorDirichletMethods:
    """
    Tests the methods of the TensorDirichlet distribution.

    This suite verifies that:
    - `sample` and `rsample` produce tensors of the correct shape that sum to 1.
    - `log_prob` computes the correct log probability.
    - `mean` and `entropy` match expected values.
    """

    @pytest.fixture
    def dist(self):
        """Provides a standard TensorDirichlet distribution."""
        concentration = torch.tensor([0.5, 1.0, 2.0])
        return TensorDirichlet(
            concentration=concentration,
            shape=concentration.shape,
            device=concentration.device,
        )

    def test_sample_shape_and_value(self, dist):
        """The sampled tensor should have the correct shape and sum to 1."""
        sample = dist.sample()
        assert sample.shape == dist.shape
        assert_close(sample.sum(dim=-1), torch.tensor(1.0))

        samples = dist.sample(sample_shape=torch.Size([4, 4]))
        assert samples.shape == (4, 4) + dist.shape
        assert_close(samples.sum(dim=-1), torch.ones(4, 4))

    def test_rsample_shape_and_value(self, dist):
        """The r-sampled tensor should have the correct shape and require grad."""
        dist.concentration.requires_grad = True
        rsample = dist.rsample()
        assert rsample.shape == dist.shape
        assert rsample.requires_grad
        assert_close(rsample.sum(dim=-1), torch.tensor(1.0))

    def test_log_prob(self, dist):
        """The log_prob should be consistent with the underlying torch distribution."""
        value = torch.tensor([0.1, 0.7, 0.2])
        expected_log_prob = dist.dist().log_prob(value)
        assert_close(dist.log_prob(value), expected_log_prob)

    def test_log_prob_of_invalid_value(self, dist):
        """The log_prob of a value not on the simplex should raise ValueError with validation enabled."""
        value = torch.tensor([0.1, 0.8, 0.2])  # Sums to 1.1
        with pytest.raises(
            ValueError, match="Expected value argument.*to be within the support"
        ):
            dist.log_prob(value)

    def test_mean(self, dist):
        """The mean should be the normalized concentration."""
        concentration = dist.concentration
        expected_mean = concentration / concentration.sum(-1, keepdim=True)
        assert_close(dist.mean, expected_mean)

    def test_entropy(self, dist):
        """The entropy should be consistent with the underlying torch distribution."""
        expected_entropy = dist.dist().entropy()
        assert_close(dist.entropy(), expected_entropy)
