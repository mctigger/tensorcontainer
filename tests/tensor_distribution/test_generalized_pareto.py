"""
Tests for TensorGeneralizedPareto distribution.

This module contains test classes that verify:
- TensorGeneralizedPareto initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import GeneralizedPareto as TorchGeneralizedPareto

from tensorcontainer.tensor_distribution.generalized_pareto import GeneralizedPareto
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorGeneralizedParetoInitialization:
    @pytest.mark.parametrize("scale_shape", [(1,), (5,), (3, 5), (2, 4, 5)])
    @pytest.mark.parametrize("concentration_shape", [(1,), (5,), (3, 5), (2, 4, 5)])
    def test_init_with_scale_and_concentration(self, scale_shape, concentration_shape):
        """TensorGeneralizedPareto should initialize correctly with 'scale' and 'concentration'."""
        scale = torch.rand(*scale_shape) + 0.1  # scale must be > 0
        concentration = torch.randn(*concentration_shape)
        td_gp = GeneralizedPareto(scale=scale, concentration=concentration)
        assert td_gp.batch_shape == torch.broadcast_shapes(scale_shape, concentration_shape)
        assert td_gp.event_shape == torch.Size([])


class TestTensorGeneralizedParetoTensorContainerIntegration:
    @pytest.mark.parametrize("scale_shape", [(1,), (5,), (3, 5)])
    @pytest.mark.parametrize("concentration_shape", [(1,), (5,), (3, 5)])
    def test_compile_compatibility(self, scale_shape, concentration_shape):
        """Core operations should be compatible with torch.compile."""
        scale = torch.rand(*scale_shape) + 0.1
        concentration = torch.randn(*concentration_shape)
        td_gp = GeneralizedPareto(scale=scale, concentration=concentration)
        sample = td_gp.sample()

        def sample_fn(td):
            return td.sample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_gp, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_gp, sample, fullgraph=False)


class TestTensorGeneralizedParetoAPIMatch:
    """
    Tests that the TensorGeneralizedPareto API matches the PyTorch GeneralizedPareto API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorGeneralizedPareto matches
        torch.distributions.GeneralizedPareto.
        """
        assert_init_signatures_match(
            GeneralizedPareto, TorchGeneralizedPareto
        )

    def test_properties_match(self):
        """
        Tests that the properties of TensorGeneralizedPareto match
        torch.distributions.GeneralizedPareto.
        """
        assert_properties_signatures_match(
            GeneralizedPareto, TorchGeneralizedPareto
        )

    def test_property_values_match(self):
        """
        Tests that the property values of TensorGeneralizedPareto match
        torch.distributions.GeneralizedPareto.
        """
        scale = torch.rand(3, 5) + 0.1
        concentration = torch.randn(3, 5)
        td_gp = GeneralizedPareto(scale=scale, concentration=concentration)
        assert_property_values_match(td_gp)