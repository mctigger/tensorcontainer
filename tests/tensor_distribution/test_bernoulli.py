import pytest
import torch
from rtd.tensor_distribution import TensorBernoulli


def normalize_device(dev: torch.device) -> torch.device:
    d = torch.device(dev)
    # If no index was given, fill in current_device() for CUDA, leave CPU as-is
    if d.type == "cuda" and d.index is None:
        idx = (
            torch.cuda.current_device()
        )  # e.g. 0 :contentReference[oaicite:4]{index=4}
        return torch.device(f"cuda:{idx}")
    return d


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ({"probs": 0.3, "logits": 0.1}, {}),  # both provided
    ],
)
def test_init_invalid_params(args, kwargs):
    with pytest.raises(TypeError):
        TensorBernoulli(**args, **kwargs)


def test_sample_shape_and_dtype_and_values():
    probs = torch.rand(4, 3)
    dist = TensorBernoulli(probs=probs, shape=probs.shape, reinterpreted_batch_ndims=0)
    # draw 5 i.i.d. samples
    samples = dist.sample(sample_shape=(5,))
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
        probs=probs,
        shape=probs.shape,
        reinterpreted_batch_ndims=rbn_dims,
    )
    x = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    lp = dist.log_prob(x)
    # expected via torch.distributions
    td = torch.distributions.Bernoulli(probs)
    ref = td.log_prob(x)
    if rbn_dims > 0:
        ref = ref.sum(dim=[0, 1][-rbn_dims:])
    assert lp.shape == expected_shape
    assert torch.allclose(lp, ref)


def test_device_normalization_helper():
    # internal devices cuda vs cuda:0
    a = torch.device("cuda")
    b = torch.ones(1, device="cuda").device  # cuda:0
    # they compare equal under the same normalization logic used by TensorBernoulli

    assert normalize_device(a) == normalize_device(b)
