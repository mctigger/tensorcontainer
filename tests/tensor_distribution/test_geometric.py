"""
Tests for TensorGeometric distribution.

This module contains test classes that verify:
- TensorGeometric initialization and parameter validation
- Core distribution operations (sample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import Geometric

from tensorcontainer.tensor_distribution.geometric import TensorGeometric
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorGeometricInitialization:
    def test_init_no_params_raises_error(self):
        """A RuntimeError should be raised when neither probs nor logits are provided."""
        with pytest.raises(
            RuntimeError, match="Either 'probs' or 'logits' must be provided."
        ):
            TensorGeometric(probs=None, logits=None) # type: ignore

    def test_init_both_params_raises_error(self):
        """A RuntimeError should be raised when both probs and logits are provided."""
        with pytest.raises(
            RuntimeError, match="Only one of 'probs' or 'logits' can be provided."
        ):
            TensorGeometric(probs=torch.rand(1), logits=torch.rand(1))


class TestTensorGeometricTensorContainerIntegration:
    @pytest.mark.parametrize("param_shape", [(5,), (3, 5), (2, 4, 5)])
    @pytest.mark.parametrize("param_type", ["probs", "logits"])
    def test_compile_compatibility(self, param_shape, param_type):
        """Core operations should be compatible with torch.compile."""
        if param_type == "probs":
            param = torch.rand(*param_shape, requires_grad=True)
            td_geometric = TensorGeometric(probs=param)
        else:
            param = torch.rand(*param_shape, requires_grad=True)
            td_geometric = TensorGeometric(logits=param)
        
        sample = td_geometric.sample()

        def sample_fn(td):
            return td.sample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_geometric, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_geometric, sample, fullgraph=False)


class TestTensorGeometricAPIMatch:
    """
    Tests that the TensorGeometric API matches the PyTorch Geometric API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorGeometric matches
        torch.distributions.Geometric.
        """
        assert_init_signatures_match(
            TensorGeometric, Geometric
        )

    def test_properties_match(self):
        """
        Tests that the properties of TensorGeometric match
        torch.distributions.Geometric.
        """
        assert_properties_signatures_match(
            TensorGeometric, Geometric
        )

    @pytest.mark.parametrize("param_type", ["probs", "logits"])
    def test_property_values_match(self, param_type):
        """
        Tests that the property values of TensorGeometric match
        torch.distributions.Geometric.
        """
        if param_type == "probs":
            param = torch.rand(3, 5)
            td_geometric = TensorGeometric(probs=param)
        else:
            param = torch.rand(3, 5)
            td_geometric = TensorGeometric(logits=param)
        assert_property_values_match(td_geometric)
