"""
Tests for TensorNegativeBinomial distribution.

This module contains test classes that verify:
- TensorNegativeBinomial initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import NegativeBinomial

from tensorcontainer.tensor_distribution.negative_binomial import TensorNegativeBinomial
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorNegativeBinomialInitialization:
    def test_init_no_total_count_raises_error(self):
        """A TypeError should be raised when total_count is not provided."""
        with pytest.raises(
            TypeError, match="missing 1 required positional argument: 'total_count'"
        ):
            TensorNegativeBinomial()

    def test_init_no_probs_or_logits_raises_error(self):
        """A ValueError should be raised when neither probs nor logits are provided."""
        with pytest.raises(
            ValueError, match="Either 'probs' or 'logits' must be provided."
        ):
            TensorNegativeBinomial(total_count=torch.tensor([1.0]))

    def test_init_both_probs_and_logits_raises_error(self):
        """A ValueError should be raised when both probs and logits are provided."""
        with pytest.raises(
            ValueError, match="Only one of 'probs' or 'logits' can be specified."
        ):
            TensorNegativeBinomial(
                total_count=torch.tensor([1.0]),
                probs=torch.tensor([0.5]),
                logits=torch.tensor([0.0]),
            )

    @pytest.mark.parametrize("total_count", [torch.tensor([-1.0]), torch.tensor([0.0])])
    def test_init_total_count_non_positive_raises_error(self, total_count):
        """A ValueError should be raised when total_count is not positive."""
        with pytest.raises(ValueError, match="total_count must be positive"):
            TensorNegativeBinomial(total_count=total_count, probs=torch.tensor([0.5]))

    def test_init_incompatible_shapes_raises_error(self):
        """A ValueError should be raised when total_count and probs/logits have incompatible shapes."""
        with pytest.raises(
            ValueError, match="total_count and probs/logits must have compatible shapes"
        ):
            TensorNegativeBinomial(
                total_count=torch.tensor([1.0, 2.0]), probs=torch.tensor([0.5, 0.6, 0.7])
            )


class TestTensorNegativeBinomialTensorContainerIntegration:
    @pytest.mark.parametrize("param_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, param_shape):
        """Core operations should be compatible with torch.compile."""
        total_count = torch.ones(param_shape) * 10
        probs = torch.rand(param_shape)
        td_nb = TensorNegativeBinomial(total_count=total_count, probs=probs)
        sample = td_nb.sample()

        def sample_fn(td):
            return td.sample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_nb, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_nb, sample, fullgraph=False)


class TestTensorNegativeBinomialAPIMatch:
    """
    Tests that the TensorNegativeBinomial API matches the PyTorch NegativeBinomial API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorNegativeBinomial matches
        torch.distributions.NegativeBinomial.
        """
        assert_init_signatures_match(TensorNegativeBinomial, NegativeBinomial)

    def test_properties_match(self):
        """
        Tests that the properties of TensorNegativeBinomial match
        torch.distributions.NegativeBinomial.
        """
        assert_properties_signatures_match(TensorNegativeBinomial, NegativeBinomial)

    def test_property_values_match(self):
        """
        Tests that the property values of TensorNegativeBinomial match
        torch.distributions.NegativeBinomial.
        """
        total_count = torch.tensor([10.0, 5.0])
        probs = torch.tensor([0.5, 0.8])
        td_nb = TensorNegativeBinomial(total_count=total_count, probs=probs)
        assert_property_values_match(td_nb)