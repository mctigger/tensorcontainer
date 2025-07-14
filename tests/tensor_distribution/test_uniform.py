"""
Tests for TensorUniform distribution.

This module contains test classes that verify:
- TensorUniform initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import Uniform

from tensorcontainer.tensor_distribution.uniform import TensorUniform
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorUniformInitialization:
    def test_init_no_low_raises_error(self):
        """A RuntimeError should be raised when low is not provided."""
        with pytest.raises(
            RuntimeError, match="'low' must be provided."
        ):
            TensorUniform(low=None, high=torch.tensor(1.0)) # type: ignore

    def test_init_no_high_raises_error(self):
        """A RuntimeError should be raised when high is not provided."""
        with pytest.raises(
            RuntimeError, match="'high' must be provided."
        ):
            TensorUniform(low=torch.tensor(0.0), high=None) # type: ignore

    @pytest.mark.parametrize(
        "low_shape, high_shape, expected_batch_shape",
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
    def test_broadcasting_shapes(self, low_shape, high_shape, expected_batch_shape):
        """Test that batch_shape is correctly determined by broadcasting."""
        low = torch.randn(low_shape)
        high = low + torch.rand(high_shape) + 1.0 # high must be greater than low
        td_uniform = TensorUniform(low=low, high=high)
        assert td_uniform.batch_shape == expected_batch_shape
        assert td_uniform.dist().batch_shape == expected_batch_shape


class TestTensorUniformTensorContainerIntegration:
    @pytest.mark.parametrize("param_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, param_shape):
        """Core operations should be compatible with torch.compile."""
        low = torch.randn(*param_shape)
        high = low + torch.rand(*param_shape) + 1.0 # high must be greater than low
        td_uniform = TensorUniform(low=low, high=high)
        
        sample = td_uniform.sample()
        rsample = td_uniform.rsample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_uniform, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_uniform, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_uniform, sample, fullgraph=False)


class TestTensorUniformAPIMatch:
    """
    Tests that the TensorUniform API matches the PyTorch Uniform API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorUniform matches
        torch.distributions.Uniform.
        """
        assert_init_signatures_match(
            TensorUniform, Uniform
        )

    def test_properties_match(self):
        """
        Tests that the properties of TensorUniform match
        torch.distributions.Uniform.
        """
        assert_properties_signatures_match(
            TensorUniform, Uniform
        )

    def test_property_values_match(self):
        """
        Tests that the property values of TensorUniform match
        torch.distributions.Uniform.
        """
        low = torch.randn(3, 5)
        high = low + torch.rand(3, 5) + 1.0 # high must be greater than low
        td_uniform = TensorUniform(low=low, high=high)
        assert_property_values_match(td_uniform)