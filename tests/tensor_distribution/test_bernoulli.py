"""
Tests for TensorBernoulli distribution.

This module contains test classes that verify:
- TensorBernoulli initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import Bernoulli

from tensorcontainer.tensor_distribution.bernoulli import TensorBernoulli
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorBernoulliInitialization:
    def test_init_no_params_raises_error(self):
        """A RuntimeError should be raised when neither probs nor logits are provided."""
        with pytest.raises(
            RuntimeError, match="Either 'probs' or 'logits' must be provided."
        ):
            TensorBernoulli()

    @pytest.mark.parametrize("probs_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_init_with_probs(self, probs_shape):
        """TensorBernoulli should initialize correctly with 'probs'."""
        probs = torch.rand(*probs_shape)
        td_bernoulli = TensorBernoulli(probs=probs)
        assert td_bernoulli.batch_shape == probs_shape
        assert td_bernoulli.event_shape == torch.Size([])

    @pytest.mark.parametrize("logits_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_init_with_logits(self, logits_shape):
        """TensorBernoulli should initialize correctly with 'logits'."""
        logits = torch.randn(*logits_shape)
        td_bernoulli = TensorBernoulli(logits=logits)
        assert td_bernoulli.batch_shape == logits_shape
        assert td_bernoulli.event_shape == torch.Size([])


class TestTensorBernoulliTensorContainerIntegration:
    @pytest.mark.parametrize("logits_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, logits_shape):
        """Core operations should be compatible with torch.compile."""
        logits = torch.randn(*logits_shape, requires_grad=True)
        td_bernoulli = TensorBernoulli(logits=logits)
        sample = td_bernoulli.sample()

        def sample_fn(td):
            return td.sample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_bernoulli, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_bernoulli, sample, fullgraph=False)


class TestTensorBernoulliAPIMatch:
    """
    Tests that the TensorBernoulli API matches the PyTorch Bernoulli API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorBernoulli matches
        torch.distributions.Bernoulli.
        """
        assert_init_signatures_match(
            TensorBernoulli, Bernoulli
        )

    def test_properties_match(self):
        """
        Tests that the properties of TensorBernoulli match
        torch.distributions.Bernoulli.
        """
        assert_properties_signatures_match(
            TensorBernoulli, Bernoulli
        )

    def test_property_values_match(self):
        """
        Tests that the property values of TensorBernoulli match
        torch.distributions.Bernoulli.
        """
        logits = torch.randn(3, 5)
        td_bernoulli = TensorBernoulli(logits=logits)
        assert_property_values_match(td_bernoulli)