import pytest
import torch

from tensorcontainer.tensor_dict import TensorDict  # adjust import as needed
from tests.conftest import skipif_no_compile
from tests.tensor_dict import common
from tests.tensor_dict.common import compare_nested_dict, compute_cat_shape

nested_dict = common.nested_dict

# Define parameter sets
SHAPE_DIM_PARAMS_VALID = [
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
    ((2, 1, 2), 1),
    ((2, 1, 2), 2),
    ((2, 1, 2), -1),
    ((2, 1, 2), -3),
]

SHAPE_DIM_PARAMS_INVALID = [
    # 1D: valid dims are [-1..0], so 1 and -2 are invalid
    ((4,), 1),
    ((4,), -2),
    # 2D: valid dims are [-2..1], so 2 and -3 are invalid
    ((2, 2), 2),
    ((2, 2), -3),
    # 3D: valid dims are [-3..2], so 3 and -4 are invalid
    ((2, 1, 2), 3),
    ((2, 1, 2), -4),
]


# ——— Valid concatenation dims across several shapes ———
@pytest.mark.parametrize("shape, dim", SHAPE_DIM_PARAMS_VALID)
def test_cat_valid_eager(nested_dict, shape, dim):
    data = nested_dict(shape)
    td = TensorDict(data, shape)

    def cat_operation(tensor_dict_instance, cat_dimension):
        return torch.cat(
            [tensor_dict_instance, tensor_dict_instance], dim=cat_dimension
        )

    cat_td = cat_operation(td, dim)

    # compute expected shape
    expected_shape = compute_cat_shape(shape, dim)
    assert cat_td.shape == expected_shape

    # Compare nested structure and values
    # The lambda for comparison should always use eager torch.cat on original tensor data
    compare_nested_dict(
        data, cat_td, lambda orig_tensor: torch.cat([orig_tensor, orig_tensor], dim=dim)
    )


# ——— Error on invalid dims ———
@pytest.mark.parametrize("shape, dim", SHAPE_DIM_PARAMS_INVALID)
def test_cat_invalid_dim_raises_eager(shape, dim, nested_dict):
    td = TensorDict(nested_dict(shape), shape)

    def cat_operation(tensor_dict_instance, cat_dimension):
        # This is the operation that is expected to raise an error
        return torch.cat(
            [tensor_dict_instance, tensor_dict_instance], dim=cat_dimension
        )

    with pytest.raises(IndexError):
        cat_operation(td, dim)


@skipif_no_compile
@pytest.mark.parametrize("shape, dim", SHAPE_DIM_PARAMS_INVALID)
def test_cat_invalid_dim_raises_compile(shape, dim, nested_dict):
    td = TensorDict(nested_dict(shape), shape)

    def cat_operation(tensor_dict_instance, cat_dimension):
        # This is the operation that is expected to raise an error
        return torch.cat(
            [tensor_dict_instance, tensor_dict_instance], dim=cat_dimension
        )

    # In compile mode, errors during tracing (e.g., with fake tensors by Dynamo)
    # are often wrapped in TorchRuntimeError.
    # We compile first, then expect the error upon execution of the compiled function.
    compiled_cat_op = torch.compile(cat_operation)
    with pytest.raises(IndexError, match="Dimension out of range"):
        compiled_cat_op(td, dim)
