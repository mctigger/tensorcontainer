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
    @pytest.mark.parametrize(
        "concentration1_shape, concentration0_shape, expected_batch_shape",
        [
            ((5,), (), (5,)),
            ((), (5,), (5,)),
            ((3, 5), (5,), (3, 5)),
            ((5,), (3, 5), (3, 5)),
            ((2, 4, 5), (5,), (2, 4, 5)),
            ((5,), (2, 4, 5), (2, 4, 5)),
            ((2, 4, 5), (2, 4, 5), (2, 4, 5)),
        ],
    )
    def test_broadcasting_shapes(
        self, concentration1_shape, concentration0_shape, expected_batch_shape
    ):
        """Test that batch_shape is correctly determined by broadcasting."""
        c1_in = torch.rand(concentration1_shape) + 0.5 if concentration1_shape else 0.5
        c0_in = torch.rand(concentration0_shape) + 0.5 if concentration0_shape else 0.5
        td_beta = TensorBeta(concentration1=c1_in, concentration0=c0_in)
        assert td_beta.batch_shape == expected_batch_shape
        assert td_beta.dist().batch_shape == expected_batch_shape


class TestTensorBetaTensorContainerIntegration:
    @pytest.mark.parametrize("param_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, param_shape):
        """Core operations should be compatible with torch.compile."""
        concentration1 = torch.rand(*param_shape) + 0.5
        concentration0 = torch.rand(*param_shape) + 0.5
        td_beta = TensorBeta(
            concentration1=concentration1, concentration0=concentration0
        )

        sample = td_beta.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_beta, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_beta, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_beta, sample, fullgraph=False)

    def test_pytree_integration(self):
        """
        We use the copy method as a proxy to ensure pytree integration (e.g. unflattening)
        works correctly.
        """
        # Test with tensor and float
        c1 = torch.rand(3, 5) + 0.5
        c0 = 0.5
        original_dist = TensorBeta(concentration1=c1, concentration0=c0)
        copied_dist = original_dist.copy()

        assert copied_dist is not original_dist
        assert isinstance(copied_dist, TensorBeta)
        torch.testing.assert_close(
            copied_dist.concentration1, original_dist.concentration1
        )
        torch.testing.assert_close(
            copied_dist.concentration0, original_dist.concentration0
        )
        assert copied_dist.device == original_dist.device


class TestTensorBetaAPIMatch:
    """
    Tests that the TensorBeta API matches the PyTorch Beta API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorBeta matches
        torch.distributions.Beta.
        """
        assert_init_signatures_match(TensorBeta, Beta)

    def test_properties_match(self):
        """
        Tests that the properties of TensorBeta match
        torch.distributions.Beta.
        """
        assert_properties_signatures_match(TensorBeta, Beta)

    def test_property_values_match(self):
        """
        Tests that the property values of TensorBeta match
        torch.distributions.Beta.
        """
        concentration1 = torch.rand(3, 5) + 0.5
        concentration0 = torch.rand(3, 5) + 0.5
        td_beta = TensorBeta(
            concentration1=concentration1, concentration0=concentration0
        )
        assert_property_values_match(td_beta)
