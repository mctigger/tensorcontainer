# tests/test_tensordict_compile.py

import pytest
import torch
import torch.utils._pytree as pytree

# Adjust the import path to match your project structure
from rtd.tensor_dict import TensorDict


# Helper function for comprehensive TensorDict comparison
def assert_td_equal(td_a: TensorDict, td_b: TensorDict):
    """
    Asserts that two TensorDicts are equal in shape, device, structure, and values.
    """
    assert td_a.shape == td_b.shape, "Shape mismatch"
    assert td_a.device == td_b.device, "Device mismatch"

    leaves_a, spec_a = pytree.tree_flatten(td_a)
    leaves_b, spec_b = pytree.tree_flatten(td_b)

    assert spec_a == spec_b, "PyTree spec mismatch (keys or nesting)"

    for tensor_a, tensor_b in zip(leaves_a, leaves_b):
        assert torch.allclose(tensor_a, tensor_b), "Tensor values mismatch"

    print(f"Assertion successful for TensorDicts with shape {td_a.shape}")


# --- Fixtures for reusable TensorDicts ---


@pytest.fixture
def base_td_data():
    """Provides raw data for a simple TensorDict."""
    return {
        "obs": torch.randn(4, 5, 3, 32, 32),
        "reward": torch.randn(4, 5, 1),
    }


@pytest.fixture
def simple_td(base_td_data):
    """A simple TensorDict instance."""
    return TensorDict(base_td_data, shape=torch.Size([4, 5]))


@pytest.fixture
def nested_td():
    """A TensorDict with nested structures."""
    B, T = 2, 3
    return TensorDict(
        {
            "obs": torch.randn(B, T, 10),
            "nested": TensorDict(
                {"state": torch.randn(B, T, 5)}, shape=torch.Size([B, T])
            ),
        },
        shape=torch.Size([B, T]),
    )


# --- Test Cases ---


def test_creation_in_compiled_fn(base_td_data):
    """Tests if a TensorDict can be created inside a compiled function."""

    def fn(obs, reward):
        # Create from raw tensors
        td = TensorDict({"obs": obs, "reward": reward}, shape=torch.Size([4, 5]))
        # Simple modification
        td["reward"] = td["reward"] * 2.0
        return td

    compiled_fn = torch.compile(fn)

    obs, reward = base_td_data["obs"], base_td_data["reward"]

    eager_result = fn(obs, reward)
    compiled_result = compiled_fn(obs, reward)

    assert_td_equal(eager_result, compiled_result)


def test_set_compiled(simple_td):
    """Tests various forms of setetting values in a compiled function."""

    def fn(td):
        td["reward"] = td["reward"] * 2.0
        return td["reward"]

    compiled_fn = torch.compile(fn)

    eager_result = fn(simple_td)
    compiled_result = compiled_fn(simple_td)

    assert_td_equal(eager_result, compiled_result)


def test_stack_compiled(simple_td):
    """Tests torch.stack on TensorDicts within a compiled function."""

    td_a = simple_td
    td_b = td_a.clone()
    td_b["reward"] += 1.0  # Make it different

    def fn(d1, d2):
        return torch.stack([d1, d2], dim=0)

    compiled_fn = torch.compile(fn)

    eager_result = fn(td_a, td_b)
    compiled_result = compiled_fn(td_a, td_b)

    # Expected shape is new_dim + old_shape
    assert eager_result.shape == torch.Size([2, 4, 5])
    assert_td_equal(eager_result, compiled_result)


def test_cat_compiled(simple_td):
    """Tests torch.cat on TensorDicts within a compiled function."""

    td_a = simple_td
    td_b = td_a.clone()

    def fn(d1, d2):
        # Concatenate along the second batch dimension (dim=1)
        return torch.cat([d1, d2], dim=1)

    compiled_fn = torch.compile(fn)

    eager_result = fn(td_a, td_b)
    compiled_result = compiled_fn(td_a, td_b)

    # Shape: [4, 5+5] -> [4, 10]
    assert eager_result.shape == torch.Size([4, 10])
    assert_td_equal(eager_result, compiled_result)


@pytest.mark.parametrize(
    "index",
    [
        0,
        slice(1, 3),
        torch.tensor([0, 3]),  # advanced indexing
    ],
)
def test_indexing_compiled(simple_td, index):
    """Tests various forms of indexing within a compiled function."""

    def fn(td):
        return td[index]

    compiled_fn = torch.compile(fn)

    eager_result = fn(simple_td)
    compiled_result = compiled_fn(simple_td)

    assert_td_equal(eager_result, compiled_result)


def test_nested_td_compiled(nested_td):
    """Tests operations on a nested TensorDict inside a compiled function."""

    def fn(td):
        # Slice the nested structure
        sliced_td = td[0]
        # Modify a leaf in the nested part
        sliced_td["nested"]["state"] = sliced_td["nested"]["state"] + 5.0
        return sliced_td

    compiled_fn = torch.compile(fn)

    eager_result = fn(nested_td)
    compiled_result = compiled_fn(nested_td)

    # The shape should be the remainder of the original batch shape
    assert eager_result.shape == torch.Size([3])
    # The nested TD's shape should also be updated
    assert eager_result["nested"].shape == torch.Size([3])
    assert_td_equal(eager_result, compiled_result)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test requires a CUDA device"
)
def test_device_move_compiled(simple_td):
    """Tests moving a TensorDict to another device in a compiled function."""

    def fn(td):
        return td.to("cuda")

    compiled_fn = torch.compile(fn)

    eager_result = fn(simple_td)
    compiled_result = compiled_fn(simple_td)

    assert eager_result.device.type == "cuda"
    assert_td_equal(eager_result, compiled_result)


def test_dtype_move_compiled(simple_td):
    """Tests changing the dtype of a TensorDict in a compiled function."""

    def fn(td):
        return td.to(torch.float16)

    compiled_fn = torch.compile(fn)

    eager_result = fn(simple_td)
    compiled_result = compiled_fn(simple_td)

    # Check one of the leaves for its dtype
    assert eager_result["obs"].dtype == torch.float16
    assert_td_equal(eager_result, compiled_result)
