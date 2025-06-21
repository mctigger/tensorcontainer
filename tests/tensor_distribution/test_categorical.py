import pytest
import torch
from torch.distributions import Independent, OneHotCategorical

from rtd.tensor_distribution import TensorCategorical


@pytest.mark.parametrize(
    "batch_shape, event_shape",
    [
        ((), (5,)),
        ((3,), (5,)),
        ((2, 4), (5,)),
    ],
)
def test_tensordistribution_categorical_basic(batch_shape, event_shape):
    num_classes = event_shape[-1]
    logits = torch.randn(*batch_shape, num_classes)
    reinterpreted_batch_ndims = 1

    td_categorical = TensorCategorical(
        logits=logits,
        reinterpreted_batch_ndims=reinterpreted_batch_ndims,
        shape=logits.shape,
        device=logits.device,
    )

    # .dist property
    torch_dist = td_categorical.dist()
    assert isinstance(torch_dist, Independent)
    assert isinstance(torch_dist.base_dist, OneHotCategorical)
    assert torch_dist.batch_shape == batch_shape
    assert torch_dist.event_shape == event_shape

    # .sample()
    sample = td_categorical.sample()
    assert sample.shape == batch_shape + event_shape

    # .rsample()
    rsample = td_categorical.rsample()
    assert rsample.shape == batch_shape + event_shape
    assert rsample.requires_grad == logits.requires_grad

    # .log_prob()
    log_prob = td_categorical.log_prob(sample)
    assert log_prob.shape == batch_shape


@pytest.mark.parametrize(
    "batch_shape, event_shape, reinterpreted_batch_ndims",
    [
        ((2, 3), (4, 5), 2),
    ],
)
def test_tensordistribution_categorical_sample_event_shape(
    batch_shape, event_shape, reinterpreted_batch_ndims
):
    logits = torch.randn(*batch_shape, *event_shape)

    td_categorical = TensorCategorical(
        logits=logits,
        reinterpreted_batch_ndims=reinterpreted_batch_ndims,
        shape=logits.shape,
        device=logits.device,
    )

    # .dist property
    torch_dist = td_categorical.dist()
    assert torch_dist.batch_shape == batch_shape
    assert torch_dist.event_shape == event_shape

    # .sample()
    sample = td_categorical.sample()
    assert sample.shape == batch_shape + event_shape

    # .log_prob()
    log_prob = td_categorical.log_prob(sample)
    assert log_prob.shape == batch_shape
