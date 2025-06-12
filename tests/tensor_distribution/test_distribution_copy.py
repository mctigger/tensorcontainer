import pytest
import torch
from rtd.tensor_distribution import TensorBernoulli, TensorNormal, TensorDistribution


@pytest.mark.parametrize(
    "TDClass, init_kwargs",
    [
        # Bernoulli via probs, non‐soft
        (
            TensorBernoulli,
            {
                "probs": torch.tensor([0.1, 0.9]),
                "reinterpreted_batch_ndims": 1,
                "shape": (2,),
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
                "shape": (3,),
                "device": torch.device("cpu"),
            },
        ),
    ],
)
def test_copy_returns_same_subclass_and_preserves_properties(TDClass, init_kwargs):
    # Construct an instance of the chosen subclass
    td: TensorDistribution = TDClass(**init_kwargs)
    td_copy = td.copy()

    # The copy must be a fresh instance of the same class
    assert isinstance(td_copy, TDClass), "Copy should preserve subclass type"
    assert td_copy.__class__ is td.__class__
    assert td_copy is not td, "Copy should not be the same object"

    # Core TensorDict properties preserved
    assert td_copy.shape == td.shape
    assert td_copy.device == td.device

    # Distribution‐specific properties preserved
    assert hasattr(td_copy, "meta_data")
    assert td_copy.meta_data == td.meta_data

    # Data keys and tensor contents preserved
    assert set(td_copy.data.keys()) == set(td.data.keys())
    for key, val in td.data.items():
        copy_val = td_copy.data[key]
        assert isinstance(copy_val, torch.Tensor)
        assert torch.allclose(copy_val, val), f"Value under key '{key}' should match"


def test_copy_independent_modification_does_not_affect_original():
    # Create an original Bernoulli TensorDistribution
    orig = TensorBernoulli(
        probs=torch.tensor([0.3, 0.7]),
        reinterpreted_batch_ndims=1,
        shape=(2,),
        device=torch.device("cpu"),
    )
    cpy = orig.copy()

    # Remove a key from the copy
    key = next(iter(cpy.data.keys()))
    del cpy.data[key]
    assert key in orig.data, (
        "Deleting key on copy should not remove it from the original"
    )
    assert key not in cpy.data

    # Add a new key to the copy
    cpy.data["new_key"] = torch.zeros_like(orig.data[key])
    assert "new_key" in cpy.data
    assert "new_key" not in orig.data, "Adding to copy should not affect the original"
