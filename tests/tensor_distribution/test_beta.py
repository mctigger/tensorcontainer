"""
Tests for TensorBeta distribution.

This module contains test classes that verify:
- TensorBeta initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import Beta

from tensorcontainer.tensor_distribution.beta import TensorBeta
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorBetaInitialization:
    def test_init_no_concentration1_raises_error(self):
        """A RuntimeError should be raised when concentration1 is not provided."""
        with pytest.raises(
            RuntimeError, match="'concentration1' must be provided."
        ):
            TensorBeta(concentration1=None, concentration0=torch.tensor(1.0)) # type: ignore

    def test_init_no_concentration0_raises_error(self):
        """A RuntimeError should be raised when concentration0 is not provided."""
        with pytest.raises(
            RuntimeError, match="'concentration0' must be provided."
        ):
            TensorBeta(concentration1=torch.tensor(1.0), concentration0=None) # type: ignore

    @pytest.mark.parametrize(
        "concentration1_shape, concentration0_shape, expected_batch_shape",
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
    def test_broadcasting_shapes(self, concentration1_shape, concentration0_shape, expected_batch_shape):
        """Test that batch_shape is correctly determined by broadcasting."""
        concentration1 = torch.rand(concentration1_shape) + 0.5 # concentration must be > 0
        concentration0 = torch.rand(concentration0_shape) + 0.5 # concentration must be > 0
        td_beta = TensorBeta(concentration1=concentration1, concentration0=concentration0)
        assert td_beta.batch_shape == expected_batch_shape
        assert td_beta.dist().batch_shape == expected_batch_shape


class TestTensorBetaTensorContainerIntegration:
    @pytest.mark.parametrize("param_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, param_shape):
        """Core operations should be compatible with torch.compile."""
        concentration1 = torch.rand(*param_shape) + 0.5
        concentration0 = torch.rand(*param_shape) + 0.5
        td_beta = TensorBeta(concentration1=concentration1, concentration0=concentration0)
        
        sample = td_beta.sample()
        rsample = td_beta.rsample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_beta, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_beta, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_beta, sample, fullgraph=False)


class TestTensorBetaAPIMatch:
    """
    Tests that the TensorBeta API matches the PyTorch Beta API.
    """

    @pytest.mark.skip(reason="TensorBeta __init__ intentionally differs from torch.distributions.Beta")
    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorBeta matches
        torch.distributions.Beta.
        """
        assert_init_signatures_match(
            TensorBeta, Beta
        )

    @pytest.mark.skip(reason="TensorBeta properties intentionally differ from torch.distributions.Beta")
    def test_properties_match(self):
        """
        Tests that the properties of TensorBeta match
        torch.distributions.Beta.
        """
        assert_properties_signatures_match(
            TensorBeta, Beta
        )

    def test_property_values_match(self):
        """
        Tests that the property values of TensorBeta match
        torch.distributions.Beta.
        """
        concentration1 = torch.rand(3, 5) + 0.5
        concentration0 = torch.rand(3, 5) + 0.5
        td_beta = TensorBeta(concentration1=concentration1, concentration0=concentration0)
        assert_property_values_match(td_beta)