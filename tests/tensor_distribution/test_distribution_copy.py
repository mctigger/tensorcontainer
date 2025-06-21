import pytest
import torch

from rtd.tensor_distribution import TensorBernoulli, TensorDistribution, TensorNormal


@pytest.mark.parametrize(
    "TDClass, init_kwargs",
    [
        # Bernoulli via probs, non‐soft
        (
            TensorBernoulli,
            {
                "_probs": torch.tensor([0.1, 0.9]),
                "reinterpreted_batch_ndims": 1,
                "shape": torch.Size([2]),
                "device": torch.device("cpu"),
            },
        ),
        # Normal distribution
        (
            TensorNormal,
            {
                "loc": torch.randn(3, 2),
                "scale": torch.rand(3, 2) + 0.1,
                "reinterpreted_batch_ndims": 2,
                "shape": torch.Size([3, 2]),
                "device": torch.device("cpu"),
            },
        ),
    ],
)
def test_copy_returns_same_subclass_and_preserves_properties(TDClass, init_kwargs):
    # Construct an instance of the chosen subclass
    td: TensorDistribution = TDClass(**init_kwargs)
    td_copy = td.clone()

    # The copy must be a fresh instance of the same class
    assert isinstance(td_copy, TDClass), "Copy should preserve subclass type"
    assert td_copy.__class__ is td.__class__
    assert td_copy is not td, "Copy should not be the same object"

    # Core TensorDict properties preserved
    assert td_copy.shape == td.shape
    assert td_copy.device == td.device

    # Distribution‐specific properties preserved
    _, (_, _, _, _, meta_data) = td._pytree_flatten()
    _, (_, _, _, _, meta_data_copy) = td_copy._pytree_flatten()
    assert meta_data == meta_data_copy

    # Data keys and tensor contents preserved
    leaves, _ = td._pytree_flatten()
    leaves_copy, _ = td_copy._pytree_flatten()
    for leaf, leaf_copy in zip(leaves, leaves_copy):
        if isinstance(leaf, torch.Tensor):
            assert isinstance(leaf_copy, torch.Tensor)
            assert torch.allclose(leaf, leaf_copy)


def test_copy_independent_modification_does_not_affect_original():
    # Create an original Bernoulli TensorDistribution
    probs = torch.tensor([0.3, 0.7])
    orig = TensorBernoulli(
        _probs=probs,
        reinterpreted_batch_ndims=1,
        shape=probs.shape,
        device=probs.device,
    )
    cpy = orig.clone()

    # Remove a key from the copy
    leaves, _ = cpy._pytree_flatten()
    leaves_orig, _ = orig._pytree_flatten()
    assert len(leaves) == len(leaves_orig)
    for leaf, leaf_orig in zip(leaves, leaves_orig):
        if isinstance(leaf, torch.Tensor):
            assert isinstance(leaf_orig, torch.Tensor)
            assert torch.allclose(leaf, leaf_orig)

    # Add a new key to the copy
    leaves, _ = cpy._pytree_flatten()
    leaves_orig, _ = orig._pytree_flatten()
    assert len(leaves) == len(leaves_orig)
