import pytest
import torch
from torch.distributions import Bernoulli, Independent

from rtd.tensor_distribution import TensorBernoulli, TensorDistribution


def test_stack_tensordistribution_returns_tensordistribution():
    # Create two Bernoulli TensorDistributions with the same batch‐shape
    tb1 = TensorBernoulli(
        probs=torch.tensor([0.2, 0.8]), shape=(2,), device=torch.device("cpu")
    )
    tb2 = TensorBernoulli(
        probs=torch.tensor([0.3, 0.7]), shape=(2,), device=torch.device("cpu")
    )

    # Stack along a new leading dimension
    stacked = torch.stack([tb1, tb2], dim=0)

    # Should be a TensorDistribution, not a TensorDict
    assert isinstance(stacked, TensorDistribution)

    # The underlying torch.distribution should reflect the stacking
    dist = stacked.dist()
    base_dist = dist.base_dist
    assert isinstance(dist, Independent)
    assert isinstance(base_dist, Bernoulli)
    # Expect probs shape = (2, 2): two distributions each of length 2
    assert base_dist.probs.shape == (2, 2)

    # Check that each slice matches the originals
    assert torch.allclose(base_dist.probs[0], tb1.dist().base_dist.probs)
    assert torch.allclose(base_dist.probs[1], tb2.dist().base_dist.probs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_stack_tensordistribution_on_cuda():
    # Prepare two distributions on GPU
    tb1 = TensorBernoulli(
        probs=torch.tensor([0.4, 0.6], device="cuda"),
        shape=(2,),
        device=torch.device("cuda"),
    )
    tb2 = TensorBernoulli(
        probs=torch.tensor([0.1, 0.9], device="cuda"),
        shape=(2,),
        device=torch.device("cuda"),
    )

    # Stack them
    stacked = torch.stack([tb1, tb2], dim=0)

    # Still a TensorDistribution on CUDA
    assert isinstance(stacked, TensorDistribution)
    assert stacked.device.type == "cuda"

    # Underlying distribution has correct device and shape
    dist = stacked.dist().base_dist
    assert dist.probs.device.type == "cuda"
    assert dist.probs.shape == (2, 2)

    # Values match originals
    assert torch.allclose(dist.probs[0], tb1.dist().base_dist.probs)
    assert torch.allclose(dist.probs[1], tb2.dist().base_dist.probs)
