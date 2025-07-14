"""
Tests for TensorRelaxedBernoulli distribution.

This module contains test classes that verify:
- TensorRelaxedBernoulli initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import RelaxedBernoulli

from tensorcontainer.tensor_distribution.relaxed_bernoulli import TensorRelaxedBernoulli
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorRelaxedBernoulliInitialization:
    def test_init_no_probs_or_logits_raises_error(self):
        """A RuntimeError should be raised when neither probs nor logits are provided."""
        with pytest.raises(
            RuntimeError, match="Either `probs` or `logits` must be provided."
        ):
            TensorRelaxedBernoulli(temperature=torch.tensor(0.5))

    def test_init_temperature_and_probs(self):
        temperature = torch.tensor(0.5)
        probs = torch.tensor([0.1, 0.9])
        dist = TensorRelaxedBernoulli(temperature=temperature, probs=probs)
        assert dist._temperature is temperature
        assert dist._probs is probs
        assert dist._logits is None
        assert dist.batch_shape == probs.shape
        assert dist.device == probs.device

    def test_init_temperature_and_logits(self):
        temperature = torch.tensor(0.5)
        logits = torch.tensor([0.0, 1.0])
        dist = TensorRelaxedBernoulli(temperature=temperature, logits=logits)
        assert dist._temperature is temperature
        assert dist._probs is None
        assert dist._logits is logits
        assert dist.batch_shape == logits.shape
        assert dist.device == logits.device


class TestTensorRelaxedBernoulliTensorContainerIntegration:
    @pytest.mark.parametrize("logits_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, logits_shape):
        """Core operations should be compatible with torch.compile."""
        temperature = torch.tensor(0.5)
        logits = torch.randn(*logits_shape, requires_grad=True)
        td_relaxed_bernoulli = TensorRelaxedBernoulli(temperature=temperature, logits=logits)
        sample = td_relaxed_bernoulli.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_relaxed_bernoulli, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_relaxed_bernoulli, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_relaxed_bernoulli, sample, fullgraph=False)


class TestTensorRelaxedBernoulliAPIMatch:
    """
    Tests that the TensorRelaxedBernoulli API matches the PyTorch RelaxedBernoulli API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorRelaxedBernoulli matches
        torch.distributions.RelaxedBernoulli.
        """
        assert_init_signatures_match(
            TensorRelaxedBernoulli, RelaxedBernoulli
        )

    def test_properties_match(self):
        """
        Tests that the properties of TensorRelaxedBernoulli match
        torch.distributions.RelaxedBernoulli.
        """
        assert_properties_signatures_match(
            TensorRelaxedBernoulli, RelaxedBernoulli
        )

    def test_property_values_match(self):
        """
        Tests that the property values of TensorRelaxedBernoulli match
        torch.distributions.RelaxedBernoulli.
        """
        temperature = torch.tensor(0.5)
        logits = torch.randn(3, 5)
        td_relaxed_bernoulli = TensorRelaxedBernoulli(temperature=temperature, logits=logits)
        assert_property_values_match(td_relaxed_bernoulli)