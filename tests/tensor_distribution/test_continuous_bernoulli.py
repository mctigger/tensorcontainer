"""
Tests for TensorContinuousBernoulli distribution.

This module contains test classes that verify:
- TensorContinuousBernoulli initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import ContinuousBernoulli as TorchContinuousBernoulli

from tensorcontainer.tensor_distribution.continuous_bernoulli import ContinuousBernoulli
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorContinuousBernoulliInitialization:
    def test_init_no_params_raises_error(self):
        """A ValueError should be raised when neither probs nor logits are provided."""
        with pytest.raises(
            RuntimeError, match="Either 'probs' or 'logits' must be provided."
        ):
            ContinuousBernoulli()


class TestTensorContinuousBernoulliTensorContainerIntegration:
    @pytest.mark.parametrize("probs_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, probs_shape):
        """Core operations should be compatible with torch.compile."""
        probs = torch.rand(*probs_shape, requires_grad=True)
        td_continuous_bernoulli = ContinuousBernoulli(probs=probs)
        sample = td_continuous_bernoulli.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_continuous_bernoulli, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_continuous_bernoulli, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_continuous_bernoulli, sample, fullgraph=False)


class TestTensorContinuousBernoulliAPIMatch:
    """
    Tests that the TensorContinuousBernoulli API matches the PyTorch ContinuousBernoulli API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorContinuousBernoulli matches
        torch.distributions.ContinuousBernoulli.
        """
        assert_init_signatures_match(
            ContinuousBernoulli, TorchContinuousBernoulli
        )

    def test_properties_match(self):
        """
        Tests that the properties of TensorContinuousBernoulli match
        torch.distributions.ContinuousBernoulli.
        """
        assert_properties_signatures_match(
            ContinuousBernoulli, TorchContinuousBernoulli
        )

    def test_property_values_match(self):
        """
        Tests that the property values of TensorContinuousBernoulli match
        torch.distributions.ContinuousBernoulli.
        """
        probs = torch.rand(3, 5)
        td_cb = ContinuousBernoulli(probs=probs)
        assert_property_values_match(td_cb)