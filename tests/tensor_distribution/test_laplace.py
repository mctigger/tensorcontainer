"""
Tests for TensorLaplace distribution.

This module contains test classes that verify:
- TensorLaplace initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import Laplace

from tensorcontainer.tensor_distribution.laplace import TensorLaplace
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorLaplaceInitialization:
    @pytest.mark.parametrize(
        "loc_shape, scale_shape",
        [
            ((), ()),
            ((5,), ()),
            ((), (5,)),
            ((3, 5), (5,)),
            ((2, 4, 5), (4, 5)),
        ],
    )
    def test_init_valid_params(self, loc_shape, scale_shape):
        loc = torch.randn(loc_shape)
        scale = torch.rand(scale_shape) + 1e-3  # Ensure scale is positive
        td_laplace = TensorLaplace(loc=loc, scale=scale)
        assert td_laplace._loc.shape == loc.shape
        assert td_laplace._scale.shape == scale.shape

    def test_init_invalid_scale_raises_error(self):
        """A ValueError should be raised when scale is not positive."""
        loc = torch.randn(())
        scale = torch.tensor(-1.0)
        with pytest.raises(ValueError, match="scale must be positive"):
            TensorLaplace(loc=loc, scale=scale)

    def test_init_incompatible_shapes_raises_error(self):
        """A ValueError should be raised when loc and scale have incompatible shapes."""
        loc = torch.randn(2, 3)
        scale = torch.rand(4, 5) + 1e-3 # Ensure scale is positive
        with pytest.raises(ValueError, match="loc and scale must have compatible shapes"):
            TensorLaplace(loc=loc, scale=scale)


class TestTensorLaplaceTensorContainerIntegration:
    @pytest.mark.parametrize("loc_shape, scale_shape", [((5,), (5,)), ((3, 5), (5,)), ((2, 4, 5), (4, 5))])
    def test_compile_compatibility(self, loc_shape, scale_shape):
        """Core operations should be compatible with torch.compile."""
        loc = torch.randn(loc_shape, requires_grad=True)
        scale = torch.rand(scale_shape, requires_grad=True) + 1e-3
        td_laplace = TensorLaplace(loc=loc, scale=scale)
        sample = td_laplace.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_laplace, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_laplace, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_laplace, sample, fullgraph=False)


class TestTensorLaplaceAPIMatch:
    """
    Tests that the TensorLaplace API matches the PyTorch Laplace API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorLaplace matches
        torch.distributions.Laplace.
        """
        assert_init_signatures_match(TensorLaplace, Laplace)

    def test_properties_match(self):
        """
        Tests that the properties of TensorLaplace match
        torch.distributions.Laplace.
        """
        assert_properties_signatures_match(TensorLaplace, Laplace)

    def test_property_values_match(self):
        """
        Tests that the property values of TensorLaplace match
        torch.distributions.Laplace.
        """
        loc = torch.randn(3, 5)
        scale = torch.rand(3, 5) + 1e-3
        td_laplace = TensorLaplace(loc=loc, scale=scale)
        assert_property_values_match(td_laplace)