"""
Tests for TensorRelaxedCategorical distribution.

This module contains test classes that verify:
- TensorRelaxedCategorical initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import RelaxedOneHotCategorical

from tensorcontainer.tensor_distribution.relaxed_categorical import (
    TensorRelaxedCategorical,
)
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorRelaxedCategoricalInitialization:
    def test_init_no_params_raises_error(self):
        """A RuntimeError should be raised when neither probs nor logits are provided."""
        with pytest.raises(
            RuntimeError, match="Either 'probs' or 'logits' must be provided."
        ):
            TensorRelaxedCategorical(temperature=torch.tensor(1.0))

    def test_init_no_temperature_raises_error(self):
        """A TypeError should be raised when temperature is not provided."""
        with pytest.raises(
            TypeError, match="missing 1 required positional argument: 'temperature'"
        ):
            TensorRelaxedCategorical(probs=torch.randn(3, 5))

    @pytest.mark.parametrize("logits_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_init_valid_params(self, logits_shape):
        """Test valid initialization with temperature and logits."""
        temperature = torch.tensor(1.0)
        logits = torch.randn(*logits_shape)
        td_relaxed_cat = TensorRelaxedCategorical(temperature=temperature, logits=logits)
        assert td_relaxed_cat._temperature is temperature
        assert td_relaxed_cat._logits is logits
        assert td_relaxed_cat._probs is None
        assert td_relaxed_cat.batch_shape == logits_shape[:-1]
        assert td_relaxed_cat.event_shape == logits_shape[-1:]

    @pytest.mark.parametrize("probs_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_init_valid_probs(self, probs_shape):
        """Test valid initialization with temperature and probs."""
        temperature = torch.tensor(1.0)
        probs = torch.rand(*probs_shape)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        td_relaxed_cat = TensorRelaxedCategorical(temperature=temperature, probs=probs)
        assert td_relaxed_cat._temperature is temperature
        assert td_relaxed_cat._probs is probs
        assert td_relaxed_cat._logits is None
        assert td_relaxed_cat.batch_shape == probs_shape[:-1]
        assert td_relaxed_cat.event_shape == probs_shape[-1:]


class TestTensorRelaxedCategoricalTensorContainerIntegration:
    @pytest.mark.parametrize("logits_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, logits_shape):
        """Core operations should be compatible with torch.compile."""
        temperature = torch.tensor(1.0)
        logits = torch.randn(*logits_shape, requires_grad=True)
        td_relaxed_cat = TensorRelaxedCategorical(temperature=temperature, logits=logits)
        sample = td_relaxed_cat.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_relaxed_cat, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_relaxed_cat, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_relaxed_cat, sample, fullgraph=False)


class TestTensorRelaxedCategoricalAPIMatch:
    """
    Tests that the TensorRelaxedCategorical API matches the PyTorch RelaxedOneHotCategorical API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorRelaxedCategorical matches
        torch.distributions.RelaxedOneHotCategorical.
        """
        assert_init_signatures_match(
            TensorRelaxedCategorical, RelaxedOneHotCategorical
        )

    def test_properties_match(self):
        """
        Tests that the properties of TensorRelaxedCategorical match
        torch.distributions.RelaxedOneHotCategorical.
        """
        assert_properties_signatures_match(
            TensorRelaxedCategorical, RelaxedOneHotCategorical
        )

    def test_property_values_match(self):
        """
        Tests that the property values of TensorRelaxedCategorical match
        torch.distributions.RelaxedOneHotCategorical.
        """
        temperature = torch.tensor(1.0)
        logits = torch.randn(3, 5)
        td_relaxed_cat = TensorRelaxedCategorical(temperature=temperature, logits=logits)
        assert_property_values_match(td_relaxed_cat)