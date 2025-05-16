import torch
import pytest
from torch.distributions import Normal, Independent

from rtd.tensor_dict import TensorDict
from rtd.tensor_distribution import TensorNormal, TensorDistribution


@pytest.fixture
def tensor_normal():
    loc = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    scale = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    return TensorNormal(loc, scale, reinterpreted_batch_ndims=1, shape=[2])


def test_tensordict_access(tensor_normal):
    assert isinstance(tensor_normal, TensorDict)
    assert "loc" in tensor_normal
    assert "scale" in tensor_normal
    assert tensor_normal["loc"].shape == (2, 2)


def test_dist_method(tensor_normal):
    dist = tensor_normal.dist()
    assert isinstance(dist, Independent)
    assert isinstance(dist.base_dist, Normal)
    assert dist.base_dist.loc.shape == (2, 2)


def test_sample_from_dist(tensor_normal):
    sample = tensor_normal.dist().sample()
    assert sample.shape == (2, 2)


def test_clone(tensor_normal):
    clone = tensor_normal.clone()
    assert isinstance(clone, TensorNormal)
    assert torch.equal(clone["loc"], tensor_normal["loc"])


def test_to_device(tensor_normal):
    if torch.cuda.is_available():
        device_td = tensor_normal.to("cuda")
        assert device_td["loc"].device.type == "cuda"


def test_indexing(tensor_normal):
    sliced = tensor_normal[0]
    assert isinstance(sliced, TensorNormal)
    assert sliced["loc"].shape == (2,)


def test_view(tensor_normal):
    viewed = tensor_normal.view(1, 2)
    assert viewed["loc"].shape == (1, 2, 2)


def test_some_function(tensor_normal):
    tn = tensor_normal[None]
    sample = tn.dist().sample()
    value = tn.dist().log_prob(sample)

    def fn(tensor_distribution: TensorDistribution):
        d = tensor_distribution.dist()
        log_prob = d.log_prob(sample)  # direct access

        return log_prob

    compiled_fn = torch.compile(fn)
    out = compiled_fn(tn)

    assert isinstance(out, torch.Tensor)
    assert torch.allclose(out, value)
