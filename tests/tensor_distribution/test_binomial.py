"""
Tests for TensorBinomial distribution.

This module contains test classes that verify:
- TensorBinomial initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing

from tensorcontainer.tensor_distribution.binomial import TensorBinomial
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorBinomialInitialization:
    def test_init_no_params_raises_error(self):
        """A ValueError should be raised when neither probs nor logits are provided."""
        with pytest.raises(
            ValueError, match="Either `probs` or `logits` must be specified, but not both."
        ):
            TensorBinomial(total_count=10)

    def test_init_with_probs(self):
        """Test initialization with probs."""
        dist = TensorBinomial(total_count=10, probs=torch.tensor([0.5]))
        assert isinstance(dist, TensorBinomial)
        assert dist.total_count == 10
        torch.testing.assert_close(dist.probs, torch.tensor([0.5]))

    def test_init_with_logits(self):
        """Test initialization with logits."""
        dist = TensorBinomial(total_count=10, logits=torch.tensor([0.0]))
        assert isinstance(dist, TensorBinomial)
        assert dist.total_count == 10
        torch.testing.assert_close(dist.logits, torch.tensor([0.0]))


class TestTensorBinomialTensorContainerIntegration:
    @pytest.mark.parametrize("param_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, param_shape):
        """Core operations should be compatible with torch.compile."""
        total_count = 10
        probs = torch.rand(*param_shape, requires_grad=True)
        td_binomial = TensorBinomial(total_count=total_count, probs=probs)
        sample = td_binomial.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            # Binomial does not support rsample
            with pytest.raises(NotImplementedError):
                return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_binomial, fullgraph=False)
        # run_and_compare_compiled(rsample_fn, td_binomial, fullgraph=False) # rsample not supported
        run_and_compare_compiled(log_prob_fn, td_binomial, sample, fullgraph=False)


class TestTensorBinomialAPIMatch:
    """
    Tests that the TensorBinomial API matches the PyTorch Binomial API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorBinomial matches
        torch.distributions.Binomial.
        """
        assert_init_signatures_match(
            TensorBinomial, torch.distributions.Binomial
        )

    def test_properties_match(self):
        """
        Tests that the properties of TensorBinomial match
        torch.distributions.Binomial.
        """
        assert_properties_signatures_match(
            TensorBinomial, torch.distributions.Binomial
        )

    def test_property_values_match(self):
        """
        Tests that the property values of TensorBinomial match
        torch.distributions.Binomial.
        """
        total_count = 10
        probs = torch.rand(3, 5)
        td_binomial = TensorBinomial(total_count=total_count, probs=probs)
        assert_property_values_match(td_binomial)