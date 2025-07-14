"""
Tests for TensorVonMises distribution.

This module contains test classes that verify:
- TensorVonMises initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import VonMises as TorchVonMises

from tensorcontainer.tensor_distribution.von_mises import TensorVonMises
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorVonMisesInitialization:
    def test_init_no_loc_raises_error(self):
        """A RuntimeError should be raised when loc is not provided."""
        with pytest.raises(
            RuntimeError, match="'loc' must be provided."
        ):
            TensorVonMises(loc=None, concentration=torch.tensor([1.0]))

    def test_init_no_concentration_raises_error(self):
        """A RuntimeError should be raised when concentration is not provided."""
        with pytest.raises(
            RuntimeError, match="'concentration' must be provided."
        ):
            TensorVonMises(loc=torch.tensor([0.0]), concentration=None)


class TestTensorVonMisesTensorContainerIntegration:
    @pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, shape):
        """Core operations should be compatible with torch.compile."""
        loc = torch.randn(*shape, requires_grad=True)
        concentration = torch.rand(*shape, requires_grad=True) + 0.1 # concentration must be > 0
        td_von_mises = TensorVonMises(loc=loc, concentration=concentration)
        sample = td_von_mises.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_von_mises, fullgraph=False)
        # VonMises does not support rsample
        # run_and_compare_compiled(rsample_fn, td_von_mises, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_von_mises, sample, fullgraph=False)


class TestTensorVonMisesAPIMatch:
    """
    Tests that the TensorVonMises API matches the PyTorch VonMises API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorVonMises matches
        torch.distributions.VonMises.
        """
        assert_init_signatures_match(
            TensorVonMises, TorchVonMises
        )

    def test_properties_match(self):
        """
        Tests that the properties of TensorVonMises match
        torch.distributions.VonMises.
        """
        assert_properties_signatures_match(
            TensorVonMises, TorchVonMises
        )

    def test_property_values_match(self):
        """
        Tests that the property values of TensorVonMises match
        torch.distributions.VonMises.
        """
        loc = torch.randn(3, 5)
        concentration = torch.rand(3, 5) + 0.1 # concentration must be > 0
        td_von_mises = TensorVonMises(loc=loc, concentration=concentration)
        assert_property_values_match(td_von_mises)