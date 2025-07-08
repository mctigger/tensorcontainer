import pytest
import torch

from tensorcontainer.tensor_distribution.bernoulli import TensorBernoulli
from tests.conftest import skipif_no_cuda

from .conftest import normalize_device


@pytest.mark.parametrize(
    "args,kwargs",
    [
        (
            {"_probs": torch.tensor(0.3), "_logits": torch.tensor(0.1)},
            {},
        ),  # both provided
    ],
)
def test_init_invalid_params(args, kwargs):
    with pytest.raises(ValueError):
        TensorBernoulli(shape=(), device=torch.device("cpu"), **args, **kwargs)


def test_sample_shape_and_dtype_and_values():
    probs = torch.rand(4, 3)
    dist = TensorBernoulli(
        _probs=probs,
        reinterpreted_batch_ndims=0,
        shape=probs.shape,
        device=probs.device,
    )
    # draw 5 i.i.d. samples
    samples = dist.sample(sample_shape=torch.Size([5]))
    # shape = (5, *batch_shape)
    assert samples.shape == (5, *probs.shape)
    assert samples.dtype == torch.float32  # Bernoulli returns floats 0/1
    # values must be 0 or 1
    assert ((samples == 0) | (samples == 1)).all()


@pytest.mark.parametrize(
    "rbn_dims,expected_shape",
    [
        (0, (2, 3)),  # no reinterpret → log_prob per-element
        (1, (2,)),  # sum over last 1 dim
        (2, ()),  # sum over last 2 dims → scalar
    ],
)
def test_log_prob_reinterpreted_batch_ndims(rbn_dims, expected_shape):
    probs = torch.tensor([[0.2, 0.8, 0.1], [0.5, 0.5, 0.5]])
    dist = TensorBernoulli(
        _probs=probs,
        reinterpreted_batch_ndims=rbn_dims,
        shape=probs.shape,
        device=probs.device,
    )
    x = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    lp = dist.log_prob(x)
    # expected via torch.distributions
    td = torch.distributions.Bernoulli(probs)
    ref = td.log_prob(x)
    if rbn_dims > 0:
        ref = ref.sum(dim=list(range(len(ref.shape)))[-rbn_dims:])
    assert lp.shape == expected_shape
    assert torch.allclose(lp, ref)


@skipif_no_cuda
def test_device_normalization_helper():
    # internal devices cuda vs cuda:0
    a = torch.device("cuda")
    b = torch.ones(1, device="cuda").device  # cuda:0
    # they compare equal under the same normalization logic used by TensorBernoulli

    assert normalize_device(a) == normalize_device(b)


def test_init_logits():
    logits = torch.randn(4, 3)
    dist = TensorBernoulli(_logits=logits, shape=logits.shape, device=logits.device)
    assert torch.allclose(dist.logits, logits)
    assert torch.allclose(dist.probs, torch.sigmoid(logits))


def test_sample_logits():
    logits = torch.randn(4, 3)
    dist = TensorBernoulli(
        _logits=logits,
        reinterpreted_batch_ndims=0,
        shape=logits.shape,
        device=logits.device,
    )
    samples = dist.sample(sample_shape=torch.Size([5]))
    assert samples.shape == (5, *logits.shape)
    assert samples.dtype == torch.float32
    assert ((samples == 0) | (samples == 1)).all()


def test_log_prob_logits():
    logits = torch.randn(2, 3)
    dist = TensorBernoulli(
        _logits=logits,
        reinterpreted_batch_ndims=0,
        shape=logits.shape,
        device=logits.device,
    )
    x = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    lp = dist.log_prob(x)
    td = torch.distributions.Bernoulli(logits=logits)
    ref = td.log_prob(x)
    assert torch.allclose(lp, ref)


@pytest.mark.parametrize("shape", [(4,), (2, 2)])
def test_view_probs(shape):
    probs = torch.rand(*shape)
    dist = TensorBernoulli(_probs=probs, shape=probs.shape, device=probs.device)
    dist_view = dist.view(-1)
    assert dist_view.probs.shape == (probs.numel(),)


@pytest.mark.parametrize("shape", [(4,), (2, 2)])
def test_view_logits(shape):
    logits = torch.randn(*shape)
    dist = TensorBernoulli(_logits=logits, shape=logits.shape, device=logits.device)
    dist_view = dist.view(-1)
    assert dist_view.logits.shape == (logits.numel(),)
