import pytest
import torch
from src.tensorcontainer.tensor_distribution.truncated_normal import (
    TensorTruncatedNormal,
)
from src.tensorcontainer.distributions.truncated_normal import TruncatedNormal
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)
from tests.compile_utils import run_and_compare_compiled


@pytest.mark.parametrize("batch_shape", [(), (1,), (2, 3)])
@pytest.mark.parametrize("event_shape", [(), (1,), (2, 3)])
def test_truncated_normal_init(batch_shape, event_shape, device):
    # Ensure that the shapes are correctly handled for scalar tensors
    loc_param_shape = batch_shape + event_shape
    loc = torch.randn(loc_param_shape if loc_param_shape else (), device=device)
    scale = torch.rand(loc_param_shape if loc_param_shape else (), device=device) + 1e-6
    low = torch.randn(loc_param_shape if loc_param_shape else (), device=device)
    high = (
        low
        + torch.rand(loc_param_shape if loc_param_shape else (), device=device)
        + 1e-6
    )

    dist = TensorTruncatedNormal(loc=loc, scale=scale, low=low, high=high)

    expected_batch_shape = loc.shape[:-1] if loc.ndim > 0 else torch.Size([])
    expected_event_shape = loc.shape[-1:] if loc.ndim > 0 else torch.Size([])

    assert dist.batch_shape == expected_batch_shape
    assert dist.event_shape == expected_event_shape
    assert dist.loc.shape == loc.shape
    assert dist.scale.shape == scale.shape
    assert dist.low.shape == low.shape
    assert dist.high.shape == high.shape


@pytest.mark.parametrize("batch_shape", [(), (1,), (2, 3)])
@pytest.mark.parametrize("event_shape", [(), (1,), (2, 3)])
def test_truncated_normal_api(batch_shape, event_shape, device):
    loc_param_shape = batch_shape + event_shape
    loc = torch.randn(loc_param_shape if loc_param_shape else (), device=device)
    scale = torch.rand(loc_param_shape if loc_param_shape else (), device=device) + 1e-6
    low = torch.randn(loc_param_shape if loc_param_shape else (), device=device)
    high = (
        low
        + torch.rand(loc_param_shape if loc_param_shape else (), device=device)
        + 1e-6
    )

    dist = TensorTruncatedNormal(loc=loc, scale=scale, low=low, high=high)

    assert_init_signatures_match(TensorTruncatedNormal, TruncatedNormal)
    assert_properties_signatures_match(TensorTruncatedNormal, TruncatedNormal)
    assert_property_values_match(dist)


@pytest.mark.parametrize("batch_shape", [(), (1,), (2, 3)])
@pytest.mark.parametrize("event_shape", [(), (1,), (2, 3)])
def test_truncated_normal_methods(batch_shape, event_shape, device):
    loc_param_shape = batch_shape + event_shape
    loc = torch.randn(loc_param_shape if loc_param_shape else (), device=device)
    scale = torch.rand(loc_param_shape if loc_param_shape else (), device=device) + 1e-6
    low = torch.randn(loc_param_shape if loc_param_shape else (), device=device)
    high = (
        low
        + torch.rand(loc_param_shape if loc_param_shape else (), device=device)
        + 1e-6
    )

    dist = TensorTruncatedNormal(loc=loc, scale=scale, low=low, high=high)

    # Test rsample
    sample = dist.rsample()
    assert sample.shape == dist.batch_shape + dist.event_shape

    # Test log_prob
    log_prob = dist.log_prob(sample)
    assert log_prob.shape == dist.batch_shape


@pytest.mark.parametrize("batch_shape", [(), (1,), (2, 3)])
@pytest.mark.parametrize("event_shape", [(), (1,), (2, 3)])
def test_truncated_normal_compile(batch_shape, event_shape, device):
    loc_param_shape = batch_shape + event_shape
    loc = torch.randn(loc_param_shape if loc_param_shape else (), device=device)
    scale = torch.rand(loc_param_shape if loc_param_shape else (), device=device) + 1e-6
    low = torch.randn(loc_param_shape if loc_param_shape else (), device=device)
    high = (
        low
        + torch.rand(loc_param_shape if loc_param_shape else (), device=device)
        + 1e-6
    )

    dist = TensorTruncatedNormal(
        loc=loc, scale=scale, low=low, high=high, validate_args=False
    )

    # Test rsample
    run_and_compare_compiled(dist.rsample, torch.Size((5,)))

    # Test log_prob
    value = dist.rsample(torch.Size((1,)))
    run_and_compare_compiled(dist.log_prob, value)
