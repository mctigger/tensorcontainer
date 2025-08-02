"""
Tests for TensorTruncatedNormal distribution.

This module contains test classes that verify:
- TensorTruncatedNormal initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch

from src.tensorcontainer.distributions.truncated_normal import TruncatedNormal
from src.tensorcontainer.tensor_distribution.truncated_normal import (
    TensorTruncatedNormal,
)
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


@pytest.fixture
def truncated_normal_params_factory(device):
    """
    Factory fixture that returns a function to create parameters with custom shapes.

    Args:
        device: torch device to create tensors on

    Returns:
        function: A function that takes a shape and returns parameter dict
    """

    def create_params(shape):
        loc = torch.randn(shape, device=device)
        scale = torch.rand(shape, device=device) + 1e-6
        low = torch.randn(shape, device=device)
        high = low + torch.rand(shape, device=device) + 1e-6
        return {"loc": loc, "scale": scale, "low": low, "high": high}

    return create_params


class TestTensorTruncatedNormalInitialization:
    @pytest.mark.parametrize("batch_shape", [(), (1,), (2, 3)])
    @pytest.mark.parametrize("event_shape", [(), (1,), (2, 3)])
    def test_truncated_normal_init(
        self, batch_shape, event_shape, truncated_normal_params_factory
    ):
        # Create parameters with the specific shape for this test
        loc_param_shape = batch_shape + event_shape
        params = truncated_normal_params_factory(loc_param_shape)

        dist = TensorTruncatedNormal(**params)

        expected_batch_shape = params["loc"].shape
        expected_event_shape = torch.Size([])

        assert dist.batch_shape == expected_batch_shape
        assert dist.event_shape == expected_event_shape
        assert dist.loc.shape == params["loc"].shape
        assert dist.scale.shape == params["scale"].shape
        assert dist.low.shape == params["low"].shape
        assert dist.high.shape == params["high"].shape


class TestTensorTruncatedNormalAPIMatch:
    def test_init_signatures_match(self, device):
        assert_init_signatures_match(TensorTruncatedNormal, TruncatedNormal)

    def test_properties_match(self, device):
        assert_properties_signatures_match(TensorTruncatedNormal, TruncatedNormal)

    def test_property_values_match(self, device, truncated_normal_params_factory):
        # Create parameters with default shape (3, 5) that was used in the original fixture
        params = truncated_normal_params_factory((3, 5))
        td_truncated_normal = TensorTruncatedNormal(**params)
        assert_property_values_match(td_truncated_normal)


class TestTensorTruncatedNormalMethods:
    @pytest.mark.parametrize("batch_shape", [(), (1,), (2, 3)])
    @pytest.mark.parametrize("event_shape", [(), (1,), (2, 3)])
    def test_truncated_normal_methods(
        self, batch_shape, event_shape, truncated_normal_params_factory
    ):
        # Create parameters with the specific shape for this test
        loc_param_shape = batch_shape + event_shape
        params = truncated_normal_params_factory(loc_param_shape)

        dist = TensorTruncatedNormal(**params)

        # Test rsample
        sample = dist.rsample()
        assert sample.shape == dist.batch_shape + dist.event_shape

        # Test log_prob
        log_prob = dist.log_prob(sample)
        assert log_prob.shape == dist.batch_shape


class TestTensorTruncatedNormalCompileCompatibility:
    @pytest.mark.parametrize("param_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, param_shape, truncated_normal_params_factory):
        # Create parameters with the specific shape for this test
        params = truncated_normal_params_factory(param_shape)

        td_truncated_normal = TensorTruncatedNormal(**params, validate_args=False)

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(rsample_fn, td_truncated_normal, fullgraph=False)
        value = td_truncated_normal.rsample(torch.Size((1,)))
        run_and_compare_compiled(
            log_prob_fn, td_truncated_normal, value, fullgraph=False
        )


class TestTensorTruncatedNormalPyTreeIntegration:
    @pytest.mark.parametrize("batch_shape", [(), (1,), (2, 3)])
    def test_copy_pytree_integration(
        self, batch_shape, truncated_normal_params_factory
    ):
        """
        We use the copy method as a proxy to ensure pytree integration (e.g. unflattening)
        works correctly.
        """
        loc_param_shape = batch_shape
        params = truncated_normal_params_factory(loc_param_shape)

        dist = TensorTruncatedNormal(**params)

        # Test copy operation which triggers _unflatten_distribution
        dist.copy()
