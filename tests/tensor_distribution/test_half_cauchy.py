"""
Tests for TensorHalfCauchy distribution.

This module contains test classes that verify:
- TensorHalfCauchy initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import HalfCauchy as TorchHalfCauchy

from tensorcontainer.tensor_distribution.half_cauchy import TensorHalfCauchy
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorHalfCauchyInitialization:
    def test_init_no_params_raises_error(self):
        """A TypeError should be raised when scale is not provided."""
        with pytest.raises(
            TypeError, match="missing 1 required positional argument: 'scale'"
        ):
            TensorHalfCauchy()


class TestTensorHalfCauchyTensorContainerIntegration:
    @pytest.mark.parametrize("scale_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, scale_shape):
        """Core operations should be compatible with torch.compile."""
        scale = torch.rand(*scale_shape) + 0.1  # scale must be positive
        td_half_cauchy = TensorHalfCauchy(scale=scale)
        sample = td_half_cauchy.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_half_cauchy, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_half_cauchy, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_half_cauchy, sample, fullgraph=False)


class TestTensorHalfCauchyAPIMatch:
    """
    Tests that the TensorHalfCauchy API matches the PyTorch HalfCauchy API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorHalfCauchy matches
        torch.distributions.HalfCauchy.
        """
        assert_init_signatures_match(
            TensorHalfCauchy, TorchHalfCauchy
        )

    def test_properties_match(self):
        """
        Tests that the properties of TensorHalfCauchy match
        torch.distributions.HalfCauchy.
        """
        assert_properties_signatures_match(
            TensorHalfCauchy, TorchHalfCauchy
        )

    def test_property_values_match(self):
        """
        Tests that the property values of TensorHalfCauchy match
        torch.distributions.HalfCauchy.
        """
        scale = torch.rand(3, 5) + 0.1
        td_half_cauchy = TensorHalfCauchy(scale=scale)
        assert_property_values_match(td_half_cauchy)