import pytest
import torch
from torch._dynamo import exc as dynamo_exc

from tensorcontainer.tensor_dict import TensorDict
from tests.compile_utils import run_and_compare_compiled
from tests.conftest import skipif_no_compile
from tests.tensor_dict.common import compare_nested_dict, compute_stack_shape

# Define parameter sets
SHAPE_DIM_PARAMS_VALID_STACK = [
    # 1D (input ndim=1), stack dim can be 0, 1. Negative: -1 (becomes 1), -2 (becomes 0)
    ((4,), 0),
    ((4,), 1),
    ((4,), -1),
    ((4,), -2),
    # 2D (input ndim=2), stack dim can be 0, 1, 2. Negative: -1 (->2), -2 (->1), -3 (->0)
    ((2, 2), 0),
    ((2, 2), 1),
    ((2, 2), 2),
    ((2, 2), -1),
    ((2, 2), -2),
    ((2, 2), -3),
]

SHAPE_DIM_PARAMS_INVALID_STACK = [
    # 1D: valid stack dims are 0, 1 (or -1, -2). So 2 and -3 are out of range.
    ((4,), 2),
    ((4,), -3),
    # 2D: valid stack dims are 0, 1, 2 (or -1, -2, -3). So 3 and -4 are out of range.
    ((2, 2), 3),
    ((2, 2), -4),
    # 3D: valid stack dims are 0, 1, 2, 3 (or -1, -2, -3, -4). So 4 and -5 are out of range.
    ((2, 1, 2), 4),
    ((2, 1, 2), -5),
]


# ——— Valid stacking dims across several shapes ———
@pytest.mark.parametrize("shape, dim", SHAPE_DIM_PARAMS_VALID_STACK)
def test_stack_valid_eager(shape, dim):
    data = {
        "x": {
            "a": torch.rand(*shape),
            "b": torch.rand(*shape),
        },
        "y": torch.rand(*shape),
    }
    td1 = TensorDict(data, shape)
    # Create a second TensorDict with the same data
    td2 = TensorDict(data, shape)

    def stack_operation(tensor_dict_list, stack_dimension):
        return torch.stack(tensor_dict_list, dim=stack_dimension)

    stacked_td = stack_operation([td1, td2], dim)

    # compute expected shape
    expected_shape = compute_stack_shape(shape, dim, num_tensors=2)
    assert stacked_td.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {stacked_td.shape}"
    )

    # Calculate the actual dimension index in stacked_td where stacking occurred
    original_ndim = len(shape)
    if dim >= 0:
        actual_stack_dim_idx = dim
    else:  # dim < 0, for stack, it's relative to original_ndim + 1
        actual_stack_dim_idx = dim + original_ndim + 1

    # Create slicers for __getitem__
    slicers_td1 = [slice(None)] * stacked_td.ndim
    slicers_td1[actual_stack_dim_idx] = 0
    td1_slice_from_stack = stacked_td[tuple(slicers_td1)]

    slicers_td2 = [slice(None)] * stacked_td.ndim
    slicers_td2[actual_stack_dim_idx] = 1
    td2_slice_from_stack = stacked_td[tuple(slicers_td2)]

    compare_nested_dict(
        data,
        td1_slice_from_stack,
        lambda orig_tensor: orig_tensor,  # Expect the original tensor as it's a slice
    )
    compare_nested_dict(
        data,
        td2_slice_from_stack,
        lambda orig_tensor: orig_tensor,  # Expect the original tensor
    )


# ——— Error on invalid dims ———
@pytest.mark.parametrize("shape, dim", SHAPE_DIM_PARAMS_INVALID_STACK)
def test_stack_invalid_dim_raises_eager(shape, dim):
    data = {
        "x": {
            "a": torch.rand(*shape),
            "b": torch.rand(*shape),
        },
        "y": torch.rand(*shape),
    }
    td = TensorDict(data, shape)

    def stack_operation(tensor_dict_instance, stack_dimension):
        # This is the operation that is expected to raise an error
        return torch.stack(
            [tensor_dict_instance, tensor_dict_instance], dim=stack_dimension
        )

    with pytest.raises(IndexError):
        stack_operation(td, dim)


@skipif_no_compile
@pytest.mark.parametrize("shape, dim", SHAPE_DIM_PARAMS_INVALID_STACK)
def test_stack_invalid_dim_raises_compile(shape, dim):
    data = {
        "x": {
            "a": torch.rand(*shape),
            "b": torch.rand(*shape),
        },
        "y": torch.rand(*shape),
    }
    td = TensorDict(data, shape)

    def stack_operation(tensor_dict_instance, stack_dimension):
        # This is the operation that is expected to raise an error
        return torch.stack(
            [tensor_dict_instance, tensor_dict_instance], dim=stack_dimension
        )

    compiled_stack_op = torch.compile(stack_operation, fullgraph=True)
    with pytest.raises(dynamo_exc.Unsupported):
        compiled_stack_op(td, dim)


# --- Test TD creation and stacking inside torch.compile ---
@skipif_no_compile
def test_stack_td_creation_and_stack_inside_compile():
    input_shape = (
        2,
        2,
    )  # Changed to (2,2) to be compatible with arange(0,4) -> 4 elements
    stack_dim = 0  # Example dimension

    # Define the raw data that will be used inside the compiled function
    raw_data1 = {
        "x": {
            "a": torch.arange(0, 4).reshape(*input_shape),
            "b": torch.arange(4, 8).reshape(*input_shape),
        },
        "y": torch.arange(8, 12).reshape(*input_shape),
    }
    raw_data2 = raw_data1

    def create_and_stack_tds(data1, data2, shape_val, dim_val):
        td1 = TensorDict(data1, shape_val)
        td2 = TensorDict(data2, shape_val)
        return torch.stack([td1, td2], dim=dim_val)

    compiled_fn = torch.compile(create_and_stack_tds, fullgraph=True)

    # Pass raw data and parameters to the compiled function
    # Tensors within raw_data1 and raw_data2 will be captured as constants or inputs
    stacked_td_result = compiled_fn(raw_data1, raw_data2, input_shape, stack_dim)

    # Assertions
    expected_overall_shape = compute_stack_shape(input_shape, stack_dim, num_tensors=2)
    assert stacked_td_result.shape == expected_overall_shape

    # Verify content of the first stacked TensorDict
    # Calculate the actual dimension index in stacked_td where stacking occurred
    # This logic is similar to test_stack_valid, using stack_dim directly as it's positive here.
    actual_stack_dim_idx_for_slicing = stack_dim

    slicers1 = [slice(None)] * len(expected_overall_shape)
    slicers1[actual_stack_dim_idx_for_slicing] = 0
    td1_slice = stacked_td_result[tuple(slicers1)]
    compare_nested_dict(raw_data1, td1_slice, lambda x: x)

    # Verify content of the second stacked TensorDict
    slicers2 = [slice(None)] * len(expected_overall_shape)
    slicers2[actual_stack_dim_idx_for_slicing] = 1
    td2_slice = stacked_td_result[tuple(slicers2)]
    compare_nested_dict(raw_data2, td2_slice, lambda x: x)


@pytest.mark.parametrize(
    "shape, dim",
    [
        # 1D
        ((4,), 0),
        ((4,), -1),
        # 2D
        ((2, 2), 0),
        ((2, 2), 1),
        ((2, 2), -1),
        ((1, 4), 0),
        ((1, 4), 1),
        ((1, 4), -2),
        # 3D
        ((2, 1, 2), 0),
        ((2, 1, 2), 3),
        ((2, 1, 2), -1),
        ((2, 1, 2), -4),
    ],
)
@skipif_no_compile
def test_stack_valid_compiled(shape, dim):
    def stack_fn(nd, s, d):
        td = TensorDict(nd, s)
        return torch.stack([td, td], dim=d)

    data = {
        "x": {
            "a": torch.rand(*shape),
            "b": torch.rand(*shape),
        },
        "y": torch.rand(*shape),
    }
    run_and_compare_compiled(stack_fn, data, shape, dim)
