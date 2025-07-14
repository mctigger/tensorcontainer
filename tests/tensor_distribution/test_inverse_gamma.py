"""
Tests for TensorInverseGamma distribution.

This module contains test classes that verify:
- TensorInverseGamma initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import InverseGamma as TorchInverseGamma

from tensorcontainer.tensor_distribution.inverse_gamma import TensorInverseGamma
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorInverseGammaContainerIntegration:
    @pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, shape):
        """Core operations should be compatible with torch.compile."""
        concentration = torch.rand(*shape, requires_grad=True) + 1.0
        rate = torch.rand(*shape, requires_grad=True) + 1.0
        td_inverse_gamma = TensorInverseGamma(concentration=concentration, rate=rate)
        sample = td_inverse_gamma.sample()

        def sample_fn(td):
            return td.sample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_inverse_gamma, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_inverse_gamma, sample, fullgraph=False)


class TestTensorInverseGammaAPIMatch:
    """
    Tests that the TensorInverseGamma API matches the PyTorch InverseGamma API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorInverseGamma matches
        torch.distributions.InverseGamma.
        """
        assert_init_signatures_match(
            TensorInverseGamma, TorchInverseGamma
        )

    def test_properties_match(self):
        """
        Tests that the properties of TensorInverseGamma match
        torch.distributions.InverseGamma.
        """
        assert_properties_signatures_match(
            TensorInverseGamma, TorchInverseGamma
        )

    def test_property_values_match(self):
        """
        Tests that the property values of TensorInverseGamma match
        torch.distributions.InverseGamma.
        """
        concentration = torch.rand(3, 5) + 1.0
        rate = torch.rand(3, 5) + 1.0
        td_inv_gamma = TensorInverseGamma(concentration=concentration, rate=rate)
        assert_property_values_match(td_inv_gamma)