"""
Tests for TensorDirichlet distribution.

This module contains test classes that verify:
- TensorDirichlet initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import Dirichlet

from tensorcontainer.tensor_distribution.dirichlet import TensorDirichlet
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorDirichletInitialization:
    def test_init_no_params_raises_error(self):
        """A RuntimeError should be raised when concentration is not provided."""
        with pytest.raises(
            RuntimeError, match="`concentration` must be provided."
        ):
            TensorDirichlet(concentration=None) # type: ignore


class TestTensorDirichletTensorContainerIntegration:
    @pytest.mark.parametrize("concentration_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, concentration_shape):
        """Core operations should be compatible with torch.compile."""
        concentration = torch.rand(*concentration_shape).exp()
        td_dirichlet = TensorDirichlet(concentration=concentration)
        sample = td_dirichlet.sample()

        def sample_fn(td):
            return td.sample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_dirichlet, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_dirichlet, sample, fullgraph=False)


class TestTensorDirichletAPIMatch:
    """
    Tests that the TensorDirichlet API matches the PyTorch Dirichlet API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorDirichlet matches
        torch.distributions.Dirichlet.
        """
        assert_init_signatures_match(
            TensorDirichlet, Dirichlet
        )

    def test_properties_match(self):
        """
        Tests that the properties of TensorDirichlet match
        torch.distributions.Dirichlet.
        """
        assert_properties_signatures_match(
            TensorDirichlet, Dirichlet
        )

    def test_property_values_match(self):
        """
        Tests that the property values of TensorDirichlet match
        torch.distributions.Dirichlet.
        """
        concentration = torch.rand(3, 5).exp()
        td_dirichlet = TensorDirichlet(concentration=concentration)
        assert_property_values_match(td_dirichlet)
