from typing import Any, cast

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import Multinomial

from tensorcontainer.tensor_distribution.multinomial import TensorMultinomial
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorMultinomialInitialization:
    def test_init_no_params_raises_error(self):
        """A RuntimeError should be raised when neither probs nor logits are provided."""
        with pytest.raises(
            RuntimeError, match="Either 'probs' or 'logits' must be provided."
        ):
            TensorMultinomial(total_count=1)

    def test_init_both_params_raises_error(self):
        """A RuntimeError should be raised when both probs and logits are provided."""
        with pytest.raises(
            RuntimeError, match="Only one of 'probs' or 'logits' can be provided."
        ):
            TensorMultinomial(
                total_count=1,
                probs=torch.tensor([0.5, 0.5]),
                logits=torch.tensor([0.0, 0.0]),
            )

    def test_init_total_count_non_integer_raises_error(self):
        """A ValueError should be raised when total_count is not an integer."""
        with pytest.raises(ValueError, match="total_count must be a non-negative integer."):
            TensorMultinomial(total_count=cast(Any, 1.5), probs=torch.tensor([0.5, 0.5]))

    def test_init_total_count_negative_raises_error(self):
        """A ValueError should be raised when total_count is negative."""
        with pytest.raises(ValueError, match="total_count must be a non-negative integer."):
            TensorMultinomial(total_count=-1, probs=torch.tensor([0.5, 0.5]))

    @pytest.mark.parametrize("total_count", [1, 5, 10])
    @pytest.mark.parametrize("probs_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_init_with_probs(self, total_count, probs_shape):
        """Test successful initialization with probs."""
        probs = torch.rand(*probs_shape)
        probs = probs / probs.sum(dim=-1, keepdim=True)  # Normalize probabilities
        td_multinomial = TensorMultinomial(total_count=total_count, probs=probs)
        assert td_multinomial is not None
        assert td_multinomial.total_count.item() == total_count
        torch.testing.assert_close(td_multinomial.probs, probs)

    @pytest.mark.parametrize("total_count", [1, 5, 10])
    @pytest.mark.parametrize("logits_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_init_with_logits(self, total_count, logits_shape):
        """Test successful initialization with logits."""
        logits = torch.randn(*logits_shape)
        td_multinomial = TensorMultinomial(total_count=total_count, logits=logits)
        assert td_multinomial is not None
        assert td_multinomial.total_count.item() == total_count
        torch.testing.assert_close(td_multinomial.probs, torch.softmax(logits, dim=-1))


class TestTensorMultinomialTensorContainerIntegration:
    @pytest.mark.parametrize("logits_shape", [(5,), (3, 5), (2, 4, 5)])
    @pytest.mark.parametrize("total_count", [1, 5])
    def test_compile_compatibility(self, logits_shape, total_count):
        """Core operations should be compatible with torch.compile."""
        logits = torch.randn(*logits_shape, requires_grad=True)
        td_multinomial = TensorMultinomial(total_count=total_count, logits=logits)
        sample = td_multinomial.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            # Multinomial does not have rsample
            with pytest.raises(NotImplementedError):
                return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_multinomial, fullgraph=False)
        # rsample is not implemented for Multinomial, so we don't test it
        run_and_compare_compiled(log_prob_fn, td_multinomial, sample, fullgraph=False)


class TestTensorMultinomialAPIMatch:
    """
    Tests that the TensorMultinomial API matches the PyTorch Multinomial API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorMultinomial matches
        torch.distributions.Multinomial.
        """
        assert_init_signatures_match(
            TensorMultinomial, Multinomial
        )

    def test_properties_match(self):
        """
        Tests that the properties of TensorMultinomial match
        torch.distributions.Multinomial.
        """
        assert_properties_signatures_match(
            TensorMultinomial, Multinomial
        )

    def test_property_values_match(self):
        """
        Tests that the property values of TensorMultinomial match
        torch.distributions.Multinomial.
        """
        total_count = 10
        logits = torch.randn(3, 5)
        td_multinomial = TensorMultinomial(total_count=total_count, logits=logits)
        assert_property_values_match(td_multinomial)