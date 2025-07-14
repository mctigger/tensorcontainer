"""
Tests for TensorPareto distribution.

This module contains test classes that verify:
- TensorPareto initialization and parameter validation
- Core distribution operations (sample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import Pareto as TorchPareto

from tensorcontainer.tensor_distribution.pareto import TensorPareto
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorParetoInitialization:
    def test_init_no_params_raises_error(self):
        """A RuntimeError should be raised when scale or alpha are not provided."""
        with pytest.raises(
            RuntimeError, match="`scale` must be provided."
        ):
            TensorPareto(scale=None, alpha=torch.tensor(1.0))  # type: ignore
        with pytest.raises(
            RuntimeError, match="`alpha` must be provided."
        ):
            TensorPareto(scale=torch.tensor(1.0), alpha=None)  # type: ignore


class TestTensorParetoTensorContainerIntegration:
    @pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, shape):
        """Core operations should be compatible with torch.compile."""
        scale = torch.rand(*shape) + 0.1
        alpha = torch.rand(*shape) + 0.1
        td_pareto = TensorPareto(scale=scale, alpha=alpha)
        sample = td_pareto.sample()

        def sample_fn(td):
            return td.sample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_pareto, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_pareto, sample, fullgraph=False)


class TestTensorParetoAPIMatch:
    """
    Tests that the TensorPareto API matches the PyTorch Pareto API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorPareto matches
        torch.distributions.Pareto.
        """
        assert_init_signatures_match(
            TensorPareto, TorchPareto
        )

    def test_properties_match(self):
        """
        Tests that the properties of TensorPareto match
        torch.distributions.Pareto.
        """
        assert_properties_signatures_match(
            TensorPareto, TorchPareto
        )

    def test_property_values_match(self):
        """
        Tests that the property values of TensorPareto match
        torch.distributions.Pareto.
        """
        scale = torch.rand(3, 5) + 0.1
        alpha = torch.rand(3, 5) + 0.1
        td_pareto = TensorPareto(scale=scale, alpha=alpha)
        assert_property_values_match(td_pareto)