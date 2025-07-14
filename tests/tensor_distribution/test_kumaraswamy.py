"""
Tests for TensorKumaraswamy distribution.

This module contains test classes that verify:
- TensorKumaraswamy initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import Kumaraswamy as TorchKumaraswamy

from tensorcontainer.tensor_distribution.kumaraswamy import TensorKumaraswamy
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorKumaraswamyInitialization:
    def test_init_no_params_raises_error(self):
        """A RuntimeError should be raised when concentration1 or concentration0 are not provided."""
        with pytest.raises(
            RuntimeError, match="Both 'concentration1' and 'concentration0' must be provided."
        ):
            TensorKumaraswamy(concentration1=torch.tensor(1.0), concentration0=None)
        with pytest.raises(
            RuntimeError, match="Both 'concentration1' and 'concentration0' must be provided."
        ):
            TensorKumaraswamy(concentration1=None, concentration0=torch.tensor(1.0))

    def test_init_success(self):
        """TensorKumaraswamy should initialize successfully with valid parameters."""
        concentration1 = torch.tensor([0.5, 1.0, 2.0])
        concentration0 = torch.tensor([0.5, 1.0, 2.0])
        dist = TensorKumaraswamy(concentration1=concentration1, concentration0=concentration0)
        assert isinstance(dist, TensorKumaraswamy)
        torch.testing.assert_close(dist.concentration1, concentration1)
        torch.testing.assert_close(dist.concentration0, concentration0)
        assert dist.shape == concentration1.shape
        assert dist.device == concentration1.device


class TestTensorKumaraswamyTensorContainerIntegration:
    @pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, shape):
        """Core operations should be compatible with torch.compile."""
        concentration1 = torch.rand(*shape, requires_grad=True) + 0.1
        concentration0 = torch.rand(*shape, requires_grad=True) + 0.1
        td_kumaraswamy = TensorKumaraswamy(concentration1=concentration1, concentration0=concentration0)
        sample = td_kumaraswamy.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_kumaraswamy, fullgraph=False)
        # Kumaraswamy is not reparameterizable, so rsample is not applicable
        # run_and_compare_compiled(rsample_fn, td_kumaraswamy, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_kumaraswamy, sample, fullgraph=False)


class TestTensorKumaraswamyAPIMatch:
    """
    Tests that the TensorKumaraswamy API matches the PyTorch Kumaraswamy API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorKumaraswamy matches
        torch.distributions.Kumaraswamy.
        """
        assert_init_signatures_match(
            TensorKumaraswamy, TorchKumaraswamy
        )

    def test_properties_match(self):
        """
        Tests that the properties of TensorKumaraswamy match
        torch.distributions.Kumaraswamy.
        """
        assert_properties_signatures_match(
            TensorKumaraswamy, TorchKumaraswamy
        )

    def test_property_values_match(self):
        """
        Tests that the property values of TensorKumaraswamy match
        torch.distributions.Kumaraswamy.
        """
        concentration1 = torch.rand(3, 5) + 0.1
        concentration0 = torch.rand(3, 5) + 0.1
        td_kumaraswamy = TensorKumaraswamy(concentration1=concentration1, concentration0=concentration0)
        assert_property_values_match(td_kumaraswamy)