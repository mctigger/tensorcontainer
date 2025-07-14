"""
Tests for TensorHalfNormal distribution.

This module contains test classes that verify:
- TensorHalfNormal initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import HalfNormal

from tensorcontainer.tensor_distribution.half_normal import TensorHalfNormal
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorHalfNormalInitialization:
    def test_init_invalid_scale_raises_error(self):
        """A ValueError should be raised when scale is not positive."""
        with pytest.raises(ValueError, match="scale must be positive"):
            TensorHalfNormal(scale=torch.tensor([-1.0]))
        with pytest.raises(ValueError, match="scale must be positive"):
            TensorHalfNormal(scale=torch.tensor([0.0]))

    def test_init_valid_scale(self):
        """TensorHalfNormal should initialize correctly with a positive scale."""
        scale = torch.tensor([1.0, 2.0])
        dist = TensorHalfNormal(scale=scale)
        assert dist._scale is scale
        assert dist.batch_shape == scale.shape
        assert dist.event_shape == torch.Size([])


class TestTensorHalfNormalTensorContainerIntegration:
    @pytest.mark.parametrize("scale_shape", [(1,), (3, 1), (2, 4, 1)])
    def test_compile_compatibility(self, scale_shape):
        """Core operations should be compatible with torch.compile."""
        scale = torch.rand(*scale_shape) + 0.1  # Ensure scale is positive
        td_half_normal = TensorHalfNormal(scale=scale)
        sample = td_half_normal.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_half_normal, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_half_normal, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_half_normal, sample, fullgraph=False)


class TestTensorHalfNormalAPIMatch:
    """
    Tests that the TensorHalfNormal API matches the PyTorch HalfNormal API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorHalfNormal matches
        torch.distributions.HalfNormal.
        """
        assert_init_signatures_match(
            TensorHalfNormal, HalfNormal
        )

    def test_properties_match(self):
        """
        Tests that the properties of TensorHalfNormal match
        torch.distributions.HalfNormal.
        """
        assert_properties_signatures_match(
            TensorHalfNormal, HalfNormal
        )

    def test_property_values_match(self):
        """
        Tests that the property values of TensorHalfNormal match
        torch.distributions.HalfNormal.
        """
        scale = torch.rand(3, 5) + 0.1
        td_half_normal = TensorHalfNormal(scale=scale)
        assert_property_values_match(td_half_normal)