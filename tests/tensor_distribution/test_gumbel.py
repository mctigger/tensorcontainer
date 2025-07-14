"""
Tests for TensorGumbel distribution.

This module contains test classes that verify:
- TensorGumbel initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import Gumbel as TorchGumbel

from tensorcontainer.tensor_distribution.gumbel import TensorGumbel
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorGumbelInitialization:
    def test_init_no_params_raises_error(self):
        """A ValueError should be raised when neither loc nor scale are provided."""
        with pytest.raises(
            RuntimeError, match="Either 'loc' or 'scale' must be provided."
        ):
            TensorGumbel(loc=None, scale=None)


class TestTensorGumbelTensorContainerIntegration:
    @pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, shape):
        """Core operations should be compatible with torch.compile."""
        loc = torch.randn(*shape, requires_grad=True)
        scale = torch.rand(*shape, requires_grad=True) + 0.1  # scale must be positive
        td_gumbel = TensorGumbel(loc=loc, scale=scale)
        sample = td_gumbel.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_gumbel, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_gumbel, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_gumbel, sample, fullgraph=False)


class TestTensorGumbelAPIMatch:
    """
    Tests that the TensorGumbel API matches the PyTorch Gumbel API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorGumbel matches
        torch.distributions.Gumbel.
        """
        assert_init_signatures_match(
            TensorGumbel, TorchGumbel
        )

    def test_properties_match(self):
        """
        Tests that the properties of TensorGumbel match
        torch.distributions.Gumbel.
        """
        assert_properties_signatures_match(
            TensorGumbel, TorchGumbel
        )

    def test_property_values_match(self):
        """
        Tests that the property values of TensorGumbel match
        torch.distributions.Gumbel.
        """
        loc = torch.randn(3, 5)
        scale = torch.rand(3, 5) + 0.1
        td_gumbel = TensorGumbel(loc=loc, scale=scale)
        assert_property_values_match(td_gumbel)