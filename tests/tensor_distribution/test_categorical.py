"""
Tests for TensorCategorical distribution.

This module contains test classes that verify:
- TensorCategorical initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import OneHotCategoricalStraightThrough

from tensorcontainer.tensor_distribution.categorical import TensorCategorical
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorCategoricalInitialization:
    def test_init_no_params_raises_error(self):
        """A ValueError should be raised when neither probs nor logits are provided."""
        with pytest.raises(
            RuntimeError, match="Either 'probs' or 'logits' must be provided."
        ):
            TensorCategorical()


class TestTensorCategoricalTensorContainerIntegration:
    @pytest.mark.parametrize("logits_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, logits_shape):
        """Core operations should be compatible with torch.compile."""
        logits = torch.randn(*logits_shape, requires_grad=True)
        td_categorical = TensorCategorical(logits=logits)
        sample = td_categorical.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_categorical, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_categorical, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_categorical, sample, fullgraph=False)


class TestTensorCategoricalAPIMatch:
    """
    Tests that the TensorCategorical API matches the PyTorch OneHotCategoricalStraightThrough API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorCategorical matches
        torch.distributions.OneHotCategoricalStraightThrough.
        """
        assert_init_signatures_match(
            TensorCategorical, OneHotCategoricalStraightThrough
        )

    def test_properties_match(self):
        """
        Tests that the properties of TensorCategorical match
        torch.distributions.OneHotCategoricalStraightThrough.
        """
        assert_properties_signatures_match(
            TensorCategorical, OneHotCategoricalStraightThrough
        )

    def test_property_values_match(self):
        """
        Tests that the property values of TensorCategorical match
        torch.distributions.OneHotCategoricalStraightThrough.
        """
        logits = torch.randn(3, 5)
        td_cat = TensorCategorical(logits=logits)
        assert_property_values_match(td_cat)
