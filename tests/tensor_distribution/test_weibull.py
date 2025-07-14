"""
Tests for TensorWeibull distribution.

This module contains test classes that verify:
- TensorWeibull initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import Weibull as TorchWeibull

from tensorcontainer.tensor_distribution.weibull import TensorWeibull
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorWeibullInitialization:
    def test_init_no_scale_raises_error(self):
        """A ValueError should be raised when scale is not provided."""
        with pytest.raises(
            ValueError, match="`scale` must be provided."
        ):
            TensorWeibull(scale=torch.tensor([]), concentration=torch.tensor([1.0]))

    def test_init_no_concentration_raises_error(self):
        """A ValueError should be raised when concentration is not provided."""
        with pytest.raises(
            ValueError, match="`concentration` must be provided."
        ):
            TensorWeibull(scale=torch.tensor([1.0]), concentration=torch.tensor([]))

    def test_init_valid_params(self):
        """TensorWeibull should initialize successfully with valid parameters."""
        scale = torch.tensor([1.0, 2.0])
        concentration = torch.tensor([0.5, 1.5])
        dist = TensorWeibull(scale=scale, concentration=concentration)
        assert isinstance(dist, TensorWeibull)
        torch.testing.assert_close(dist.scale, scale)
        torch.testing.assert_close(dist.concentration, concentration)


class TestTensorWeibullTensorContainerIntegration:
    @pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, shape):
        """Core operations should be compatible with torch.compile."""
        scale = torch.randn(*shape).exp()
        concentration = torch.randn(*shape).exp()
        td_weibull = TensorWeibull(scale=scale, concentration=concentration)
        sample = td_weibull.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_weibull, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_weibull, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_weibull, sample, fullgraph=False)


class TestTensorWeibullAPIMatch:
    """
    Tests that the TensorWeibull API matches the PyTorch Weibull API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorWeibull matches
        torch.distributions.Weibull.
        """
        assert_init_signatures_match(
            TensorWeibull, TorchWeibull
        )

    def test_properties_match(self):
        """
        Tests that the properties of TensorWeibull match
        torch.distributions.Weibull.
        """
        assert_properties_signatures_match(
            TensorWeibull, TorchWeibull
        )

    def test_property_values_match(self):
        """
        Tests that the property values of TensorWeibull match
        torch.distributions.Weibull.
        """
        scale = torch.randn(3, 5).exp()
        concentration = torch.randn(3, 5).exp()
        td_weibull = TensorWeibull(scale=scale, concentration=concentration)
        assert_property_values_match(td_weibull)