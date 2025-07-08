import pytest
import torch
from torch.distributions import kl_divergence, Normal, Distribution

from tensorcontainer.tensor_distribution.base import TensorDistribution


# -----------------------------------------------------------------------------
# A minimal TensorDistribution stub for testing kl registration
# -----------------------------------------------------------------------------
class DummyTensorDistribution(TensorDistribution):
    def __init__(self, torch_dist: Distribution):
        # We intentionally skip calling super().__init__(), since
        # only .dist() matters for kl_divergence dispatch.
        self._torch_dist = torch_dist

    def dist(self) -> Distribution:
        return self._torch_dist


# -----------------------------------------------------------------------------
# Test KL(TensorDistribution, TensorDistribution)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "p_params, q_params",
    [
        ({"loc": 0.0, "scale": 1.0}, {"loc": 1.0, "scale": 2.0}),
        (
            {"loc": torch.tensor([0.0, 1.0]), "scale": torch.tensor([1.0, 2.0])},
            {"loc": torch.tensor([1.0, 2.0]), "scale": torch.tensor([2.0, 3.0])},
        ),
    ],
)
def test_kl_tensordist_tensordist(p_params, q_params):
    td_p = DummyTensorDistribution(Normal(**p_params))
    td_q = DummyTensorDistribution(Normal(**q_params))

    result = kl_divergence(td_p, td_q)
    expected = kl_divergence(td_p.dist(), td_q.dist())

    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, expected)


# -----------------------------------------------------------------------------
# Test KL(TensorDistribution, Distribution)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dist_ctor",
    [
        lambda: Normal(0.0, 1.5),
        lambda: Normal(torch.tensor([0.5, 1.5]), torch.tensor([2.0, 3.0])),
    ],
)
def test_kl_tensordist_distribution(dist_ctor):
    td = DummyTensorDistribution(Normal(0.0, 1.0))
    dist = dist_ctor()

    result = kl_divergence(td, dist)
    expected = kl_divergence(td.dist(), dist)

    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, expected)


# -----------------------------------------------------------------------------
# Test KL(Distribution, TensorDistribution)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dist_ctor",
    [
        lambda: Normal(0.0, 2.0),
        lambda: Normal(torch.tensor([1.0, 2.0]), torch.tensor([0.5, 1.5])),
    ],
)
def test_kl_distribution_tensordist(dist_ctor):
    dist = dist_ctor()
    td = DummyTensorDistribution(Normal(1.0, 1.0))

    result = kl_divergence(dist, td)
    expected = kl_divergence(dist, td.dist())

    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, expected)


# -----------------------------------------------------------------------------
# Test that completely unrecognized types are not silently handled
# -----------------------------------------------------------------------------
def test_unregistered_pair_raises():
    with pytest.raises(NotImplementedError):
        kl_divergence(123, "not a distribution")
