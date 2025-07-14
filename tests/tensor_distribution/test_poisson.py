"""
Tests for TensorPoisson distribution.

This module contains test classes that verify:
- TensorPoisson initialization and parameter validation
- Core distribution operations (sample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import Poisson

from tensorcontainer.tensor_distribution.poisson import TensorPoisson
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorPoissonInitialization:
    def test_init_no_params_raises_error(self):
        """A RuntimeError should be raised when rate is not provided."""
        with pytest.raises(
            RuntimeError, match="'rate' must be provided."
        ):
            TensorPoisson(rate=None) # type: ignore


class TestTensorPoissonTensorContainerIntegration:
    @pytest.mark.parametrize("rate_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, rate_shape):
        """Core operations should be compatible with torch.compile."""
        rate = torch.rand(*rate_shape, requires_grad=True) + 0.1 # rate must be positive
        td_poisson = TensorPoisson(rate=rate)
        sample = td_poisson.sample()

        def sample_fn(td):
            return td.sample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_poisson, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_poisson, sample, fullgraph=False)


class TestTensorPoissonAPIMatch:
    """
    Tests that the TensorPoisson API matches the PyTorch Poisson API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorPoisson matches
        torch.distributions.Poisson.
        """
        assert_init_signatures_match(
            TensorPoisson, Poisson
        )

    def test_properties_match(self):
        """
        Tests that the properties of TensorPoisson match
        torch.distributions.Poisson.
        """
        assert_properties_signatures_match(
            TensorPoisson, Poisson
        )

    def test_property_values_match(self):
        """
        Tests that the property values of TensorPoisson match
        torch.distributions.Poisson.
        """
        rate = torch.rand(3, 5) + 0.1 # rate must be positive
        td_poisson = TensorPoisson(rate=rate)
        assert_property_values_match(td_poisson)
