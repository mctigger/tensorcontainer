"""
Tests for TensorExponential distribution.

This module contains test classes that verify:
- TensorExponential initialization and parameter validation
- Core distribution operations (sample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import Exponential

from tensorcontainer.tensor_distribution.exponential import TensorExponential
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorExponentialInitialization:
    def test_init_no_params_raises_error(self):
        """A RuntimeError should be raised when rate is not provided."""
        with pytest.raises(
            RuntimeError, match="'rate' must be provided."
        ):
            TensorExponential(rate=None) # type: ignore


class TestTensorExponentialTensorContainerIntegration:
    @pytest.mark.parametrize("rate_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, rate_shape):
        """Core operations should be compatible with torch.compile."""
        rate = torch.rand(*rate_shape, requires_grad=True) + 0.1 # rate must be positive
        td_exponential = TensorExponential(rate=rate)
        sample = td_exponential.sample()

        def sample_fn(td):
            return td.sample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_exponential, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_exponential, sample, fullgraph=False)


class TestTensorExponentialAPIMatch:
    """
    Tests that the TensorExponential API matches the PyTorch Exponential API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorExponential matches
        torch.distributions.Exponential.
        """
        assert_init_signatures_match(
            TensorExponential, Exponential
        )

    def test_properties_match(self):
        """
        Tests that the properties of TensorExponential match
        torch.distributions.Exponential.
        """
        assert_properties_signatures_match(
            TensorExponential, Exponential
        )

    def test_property_values_match(self):
        """
        Tests that the property values of TensorExponential match
        torch.distributions.Exponential.
        """
        rate = torch.rand(3, 5) + 0.1 # rate must be positive
        td_exp = TensorExponential(rate=rate)
        assert_property_values_match(td_exp)