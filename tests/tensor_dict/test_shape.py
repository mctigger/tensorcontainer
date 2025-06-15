import pytest
import torch

from rtd.tensor_dict import TensorDict
from tests.tensor_dict.compile_utils import run_and_compare_compiled
from tests.tensor_dict import common

nested_dict = common.nested_dict


@pytest.mark.parametrize(
    "data, shape",
    [
        (
            {
                "a": torch.zeros(1, 5),
                "b": torch.ones(1, 5),
            },
            (2,),
        ),
    ],
)
def test_constructor_raises_on_incompatible_leaf_shape(data, shape):
    def constructor_fn(data, shape):
        return TensorDict(data, shape=shape)

    with pytest.raises(ValueError):
        run_and_compare_compiled(constructor_fn, data, shape)


@pytest.mark.parametrize(
    "data, shape",
    [
        (
            {
                "x": {
                    "inner": torch.randn(3, 4, 5),
                },
                "y": torch.randn(3, 4, 5),
            },
            (2, 4),
        ),
    ],
)
def test_constructor_raises_in_nested_mapping(data, shape):
    def constructor_fn(data, shape):
        return TensorDict(data, shape=shape)

    with pytest.raises(ValueError):
        run_and_compare_compiled(constructor_fn, data, shape)


@pytest.mark.parametrize(
    "data, shape",
    [
        (
            {
                "a": torch.arange(6).reshape(2, 3),
                "b": torch.zeros(2, 5),
            },
            (2,),
        ),
    ],
)
def test_constructor_accepts_flat_dict_leading_dim_only(data, shape):
    def constructor_fn(data, shape):
        return TensorDict(data, shape=shape)

    td, _ = run_and_compare_compiled(constructor_fn, data, shape)
    # shape is stored as torch.Size
    assert td.shape == torch.Size([shape[0]])
    # contents unchanged
    assert torch.equal(td["a"], data["a"])
    assert torch.equal(td["b"], data["b"])


@pytest.mark.parametrize(
    "data, shape",
    [
        (
            {
                "x": torch.arange(24).reshape(2, 3, 4),
                "y": torch.arange(12).reshape(2, 6),
            },
            (2,),
        ),
    ],
)
def test_constructor_accepts_multidimensional_leading_batch(data, shape):
    def constructor_fn(data, shape):
        return TensorDict(data, shape=shape)

    td, _ = run_and_compare_compiled(constructor_fn, data, shape)
    # shape only enforces the first dim
    assert td.shape == torch.Size([shape[0]])
    assert td["x"].shape == (shape[0], 3, 4)
    assert td["y"].shape == (shape[0], 6)


@pytest.mark.parametrize(
    "data, shape",
    [
        (
            {
                "outer": {
                    "inner_a": torch.randn(3, 4),
                    "inner_b": torch.zeros(3, 1),
                },
                "leaf": torch.ones(3, 2),
            },
            (3,),
        ),
    ],
)
def test_constructor_accepts_nested_dict(data, shape):
    def constructor_fn(data, shape):
        return TensorDict(data, shape=shape)

    td, _ = run_and_compare_compiled(constructor_fn, data, shape)
    assert td.shape == torch.Size([shape[0]])
    # nested dict structure preserved
    assert set(td.keys()) == {"outer", "leaf"}
    assert torch.equal(td["outer"]["inner_a"], data["outer"]["inner_a"])
    assert torch.equal(td["outer"]["inner_b"], data["outer"]["inner_b"])
    assert torch.equal(td["leaf"], data["leaf"])


@pytest.mark.parametrize(
    "data, shape",
    [
        (
            {"nested": TensorDict({"a": torch.arange(8).reshape(4, 2)}, shape=(4,))},
            (4,),
        ),
    ],
)
def test_constructor_accepts_tensordict_inputs(data, shape):
    base = TensorDict({"a": torch.arange(8).reshape(4, 2)}, shape=(4,))

    def constructor_fn(data_arg, shape_arg):  # Renamed args to avoid conflict
        # This function will be compiled. Only include traceable operations.
        td_internal = TensorDict({"nested": base}, shape=shape_arg)
        return td_internal

    # Run the compiled function and get the result
    td_eager, td_compiled = run_and_compare_compiled(constructor_fn, data, shape)

    # Perform assertions on the results outside the compiled function
    # Assertions for eager result
    assert isinstance(td_eager["nested"], TensorDict)
    assert td_eager["nested"].shape == torch.Size([shape[0]])
    assert torch.equal(td_eager["nested"]["a"], base["a"])

    # Assertions for compiled result (should be consistent with eager)
    assert isinstance(td_compiled["nested"], TensorDict)
    assert td_compiled["nested"].shape == torch.Size([shape[0]])
    assert torch.equal(td_compiled["nested"]["a"], base["a"])


@pytest.mark.parametrize(
    "data, shape",
    [
        (
            {
                "a": torch.randn(5, 6, 7),
                "b": torch.zeros(2),
            },
            (),
        ),
    ],
)
def test_constructor_accepts_empty_shape(data, shape):
    def constructor_fn(data, shape):
        return TensorDict(data, shape=shape)

    td, _ = run_and_compare_compiled(constructor_fn, data, shape)
    assert td.shape == torch.Size([])
    assert td["a"].shape == (5, 6, 7)
    assert td["b"].shape == (2,)


@pytest.mark.parametrize(
    "data, shape",
    [
        (
            {
                "a": torch.zeros(0, 4),
                "b": torch.zeros(
                    0,
                ),
            },
            (0,),
        ),
    ],
)
def test_constructor_accepts_zero_batch_size(data, shape):
    def constructor_fn(data, shape):
        return TensorDict(data, shape=shape)

    td, _ = run_and_compare_compiled(constructor_fn, data, shape)
    assert td.shape == torch.Size([shape[0]])
    assert td["a"].shape == (0, 4)
    assert td["b"].shape == (0,)


@pytest.mark.parametrize(
    "data, shape",
    [
        # leaf too many dims
        ({"a": torch.randn(2, 3)}, (2, 3, 1)),
        # nested leaf mismatch
        (
            {
                "x": {
                    "y": torch.randn(
                        5,
                    )
                },
                "z": torch.randn(
                    5,
                ),
            },
            (4,),
        ),
    ],
)
def test_constructor_raises_on_shape_too_long(data, shape):
    def constructor_fn(data, shape):
        return TensorDict(data, shape=shape)

    with pytest.raises(ValueError):
        run_and_compare_compiled(constructor_fn, data, shape)
