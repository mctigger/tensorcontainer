"""
Tests for TensorChi2 distribution.

This module contains test classes that verify:
- TensorChi2 initialization and parameter validation
- Core distribution operations (sample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import Chi2

from tensorcontainer.tensor_distribution.chi2 import TensorChi2
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorChi2Initialization:
    def test_init_no_params_raises_error(self):
        """A RuntimeError should be raised when df is not provided."""
        with pytest.raises(
            RuntimeError, match="'df' must be provided."
        ):
            TensorChi2(df=None) # type: ignore


class TestTensorChi2TensorContainerIntegration:
    @pytest.mark.parametrize("param_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, param_shape):
        """Core operations should be compatible with torch.compile."""
        df = torch.rand(*param_shape).exp() + 1 # df must be positive
        td_chi2 = TensorChi2(df=df)
        
        sample = td_chi2.sample()
        rsample = td_chi2.rsample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_chi2, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_chi2, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_chi2, sample, fullgraph=False)


class TestTensorChi2APIMatch:
    """
    Tests that the TensorChi2 API matches the PyTorch Chi2 API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorChi2 matches
        torch.distributions.Chi2.
        """
        assert_init_signatures_match(
            TensorChi2, Chi2
        )

    def test_properties_match(self):
        """
        Tests that the properties of TensorChi2 match
        torch.distributions.Chi2.
        """
        assert_properties_signatures_match(
            TensorChi2, Chi2
        )

    def test_property_values_match(self):
        """
        Tests that the property values of TensorChi2 match
        torch.distributions.Chi2.
        """
        df = torch.rand(3, 5).exp() + 1 # df must be positive
        td_chi2 = TensorChi2(df=df)
        assert_property_values_match(td_chi2)