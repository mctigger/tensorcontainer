import torch

from tensorcontainer.tensor_distribution import TensorBernoulli, TensorDistribution
from tests.conftest import skipif_no_cuda


def test_stack_tensordistribution_returns_tensordistribution():
    # Create two Bernoulli TensorDistributions with the same batch‐shape
    probs1 = torch.tensor([0.2, 0.8])
    tb1 = TensorBernoulli(
        _probs=probs1,
        reinterpreted_batch_ndims=0,
        shape=probs1.shape,
        device=probs1.device,
    )
    probs2 = torch.tensor([0.3, 0.7])
    tb2 = TensorBernoulli(
        _probs=probs2,
        reinterpreted_batch_ndims=0,
        shape=probs2.shape,
        device=probs2.device,
    )

    # Stack along a new leading dimension
    stacked = torch.stack([tb1, tb2], dim=0)  # type: ignore

    # Should be a TensorDistribution, not a TensorDict
    assert isinstance(stacked, TensorDistribution)

    # The underlying torch.distribution should reflect the stacking
    # Expect probs shape = (2, 2): two distributions each of length 2
    assert stacked.shape == (2, 2)

    # Check that each slice matches the originals
    assert torch.allclose(stacked.probs[0], tb1.probs)  # type: ignore
    assert torch.allclose(stacked.probs[1], tb2.probs)  # type: ignore


@skipif_no_cuda
def test_stack_tensordistribution_on_cuda():
    # Prepare two distributions on GPU
    probs1 = torch.tensor([0.4, 0.6], device="cuda")
    tb1 = TensorBernoulli(
        _probs=probs1,
        reinterpreted_batch_ndims=0,
        shape=probs1.shape,
        device=probs1.device,
    )
    probs2 = torch.tensor([0.1, 0.9], device="cuda")
    tb2 = TensorBernoulli(
        _probs=probs2,
        reinterpreted_batch_ndims=0,
        shape=probs2.shape,
        device=probs2.device,
    )

    # Stack them
    stacked = torch.stack([tb1, tb2], dim=0)  # type: ignore

    # Still a TensorDistribution on CUDA
    assert isinstance(stacked, TensorDistribution)
    assert stacked.device.type == "cuda"

    # Underlying distribution has correct device and shape
    assert stacked.shape == (2, 2)

    # Values match originals
    assert torch.allclose(stacked.probs[0], tb1.probs)  # type: ignore
    assert torch.allclose(stacked.probs[1], tb2.probs)  # type: ignore
