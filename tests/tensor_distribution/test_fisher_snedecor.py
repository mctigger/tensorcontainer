"""
Tests for FisherSnedecor distribution.

This module contains test classes that verify:
- FisherSnedecor initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import FisherSnedecor as TorchFisherSnedecor

from tensorcontainer.tensor_distribution.fisher_snedecor import FisherSnedecor
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorFisherSnedecorInitialization:
    def test_init_no_params_raises_error(self):
        """A RuntimeError should be raised when either df1 or df2 is not provided."""
        with pytest.raises(
            RuntimeError, match="Both 'df1' and 'df2' must be provided."
        ):
            FisherSnedecor(df1=None, df2=torch.tensor([1.0]))
        with pytest.raises(
            RuntimeError, match="Both 'df1' and 'df2' must be provided."
        ):
            FisherSnedecor(df1=torch.tensor([1.0]), df2=None)


class TestTensorFisherSnedecorTensorContainerIntegration:
    @pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, shape):
        """Core operations should be compatible with torch.compile."""
        df1 = torch.rand(*shape) + 0.1
        df2 = torch.rand(*shape) + 0.1
        td_fisher_snedecor = FisherSnedecor(df1=df1, df2=df2)
        sample = td_fisher_snedecor.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_fisher_snedecor, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_fisher_snedecor, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_fisher_snedecor, sample, fullgraph=False)


class TestTensorFisherSnedecorAPIMatch:
    """
    Tests that the FisherSnedecor API matches the PyTorch FisherSnedecor API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of FisherSnedecor matches
        torch.distributions.FisherSnedecor.
        """
        assert_init_signatures_match(
            FisherSnedecor, TorchFisherSnedecor
        )

    def test_properties_match(self):
        """
        Tests that the properties of FisherSnedecor match
        torch.distributions.FisherSnedecor.
        """
        assert_properties_signatures_match(
            FisherSnedecor, TorchFisherSnedecor
        )

    def test_property_values_match(self):
        """
        Tests that the property values of FisherSnedecor match
        torch.distributions.FisherSnedecor.
        """
        df1 = torch.rand(3, 5) + 0.1
        df2 = torch.rand(3, 5) + 0.1
        td_fisher_snedecor = FisherSnedecor(df1=df1, df2=df2)
        assert_property_values_match(td_fisher_snedecor)