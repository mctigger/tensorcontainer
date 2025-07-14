"""
Tests for TensorGamma distribution.

This module contains test classes that verify:
- TensorGamma initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import Gamma

from tensorcontainer.tensor_distribution.gamma import TensorGamma
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorGammaInitialization:
    @pytest.mark.parametrize(
        "concentration_shape, rate_shape, expected_batch_shape",
        [
            ((), (), ()),
            ((5,), (), (5,)),
            ((), (5,), (5,)),
            ((3, 5), (5,), (3, 5)),
            ((5,), (3, 5), (3, 5)),
            ((2, 4, 5), (5,), (2, 4, 5)),
            ((5,), (2, 4, 5), (2, 4, 5)),
            ((2, 4, 5), (2, 4, 5), (2, 4, 5)),
        ],
    )
    def test_broadcasting_shapes(self, concentration_shape, rate_shape, expected_batch_shape):
        """Test that batch_shape is correctly determined by broadcasting."""
        concentration = torch.rand(concentration_shape) + 0.5 # concentration must be > 0
        rate = torch.rand(rate_shape) + 0.5 # rate must be > 0
        td_gamma = TensorGamma(concentration=concentration, rate=rate)
        assert td_gamma.batch_shape == expected_batch_shape
        assert td_gamma.dist().batch_shape == expected_batch_shape


class TestTensorGammaTensorContainerIntegration:
    @pytest.mark.parametrize("param_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, param_shape):
        """Core operations should be compatible with torch.compile."""
        concentration = torch.rand(*param_shape) + 0.5
        rate = torch.rand(*param_shape) + 0.5
        td_gamma = TensorGamma(concentration=concentration, rate=rate)
        
        sample = td_gamma.sample()
        rsample = td_gamma.rsample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_gamma, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_gamma, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_gamma, sample, fullgraph=False)


class TestTensorGammaAPIMatch:
    """
    Tests that the TensorGamma API matches the PyTorch Gamma API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorGamma matches
        torch.distributions.Gamma.
        """
        assert_init_signatures_match(
            TensorGamma, Gamma
        )

    def test_properties_match(self):
        """
        Tests that the properties of TensorGamma match
        torch.distributions.Gamma.
        """
        assert_properties_signatures_match(
            TensorGamma, Gamma
        )

    def test_property_values_match(self):
        """
        Tests that the property values of TensorGamma match
        torch.distributions.Gamma.
        """
        concentration = torch.rand(3, 5) + 0.5
        rate = torch.rand(3, 5) + 0.5
        td_gamma = TensorGamma(concentration=concentration, rate=rate)
        assert_property_values_match(td_gamma)