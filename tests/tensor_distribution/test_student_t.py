"""
Tests for TensorStudentT distribution.

This module contains test classes that verify:
- TensorStudentT initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import StudentT

from tensorcontainer.tensor_distribution.student_t import TensorStudentT
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorStudentTInitialization:
    @pytest.mark.parametrize(
        "df, loc, scale",
        [
            (torch.tensor(1.0), torch.tensor(0.0), torch.tensor(1.0)),
            (torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])),
            (torch.tensor([1.0]), torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])),
        ],
    )
    def test_init_valid_params(self, df, loc, scale):
        """TensorStudentT should initialize with valid parameters."""
        dist = TensorStudentT(df, loc, scale)
        assert isinstance(dist, TensorStudentT)
        assert dist.batch_shape == torch.broadcast_shapes(df.shape, loc.shape, scale.shape)

    @pytest.mark.parametrize(
        "df, loc, scale, error_msg",
        [
            (torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0), "df must be positive"),
            (torch.tensor(-1.0), torch.tensor(0.0), torch.tensor(1.0), "df must be positive"),
            (torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0), "scale must be positive"),
            (torch.tensor(1.0), torch.tensor(0.0), torch.tensor(-1.0), "scale must be positive"),
            (torch.tensor([1.0, 2.0]), torch.tensor([0.0]), torch.tensor([1.0, 1.0, 1.0]), "df, loc, and scale must have compatible shapes"),
        ],
    )
    def test_init_invalid_params_raises_error(self, df, loc, scale, error_msg):
        """A ValueError should be raised for invalid parameters."""
        with pytest.raises(ValueError, match=error_msg):
            TensorStudentT(df, loc, scale)


class TestTensorStudentTTensorContainerIntegration:
    @pytest.mark.parametrize("shape", [(1,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, shape):
        """Core operations should be compatible with torch.compile."""
        df = torch.tensor(1.0).expand(shape)
        loc = torch.tensor(0.0).expand(shape)
        scale = torch.tensor(1.0).expand(shape)
        td_student_t = TensorStudentT(df, loc, scale)
        sample = td_student_t.sample()

        def sample_fn(td):
            return td.sample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_student_t, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_student_t, sample, fullgraph=False)


class TestTensorStudentTAPIMatch:
    """
    Tests that the TensorStudentT API matches the PyTorch StudentT API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorStudentT matches
        torch.distributions.StudentT.
        """
        assert_init_signatures_match(TensorStudentT, StudentT)

    def test_properties_match(self):
        """
        Tests that the properties of TensorStudentT match
        torch.distributions.StudentT.
        """
        assert_properties_signatures_match(TensorStudentT, StudentT)

    def test_property_values_match(self):
        """
        Tests that the property values of TensorStudentT match
        torch.distributions.StudentT.
        """
        df = torch.tensor(1.0)
        loc = torch.tensor(0.0)
        scale = torch.tensor(1.0)
        td_student_t = TensorStudentT(df, loc, scale)
        assert_property_values_match(td_student_t)