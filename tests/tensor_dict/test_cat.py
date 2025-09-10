import pytest
import torch

from tensorcontainer.tensor_dict import TensorDict  # adjust import as needed
from tests.conftest import skipif_no_compile


def create_nested_dict(shape):
    a = torch.rand(*shape)
    b = torch.rand(*shape)
    y = torch.rand(*shape)
    return {"x": {"a": a, "b": b}, "y": y}


# Define parameter sets
SHAPE_DIM_PARAMS_VALID = [
    # 1D
    ((4,), (4,), 0, (8,)),
    ((4,), (4,), -1, (8,)),
    # 2D
    ((2, 2), (3, 2), 0, (5, 2)),
    ((2, 2), (2, 3), 1, (2, 5)),
    ((2, 2), (2, 3), -1, (2, 5)),
    ((2, 2), (3, 2), -2, (5, 2)),
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
@pytest.mark.parametrize("shape1, shape2, dim, expected_shape", SHAPE_DIM_PARAMS_VALID)
def test_cat_valid_eager(shape1, shape2, dim, expected_shape):
    data1 = create_nested_dict(shape1)
    data2 = create_nested_dict(shape2)

    td1 = TensorDict(data1, shape1)
    td2 = TensorDict(data2, shape2)

    cat_td = torch.cat([td1, td2], dim=dim)

    assert cat_td.shape == expected_shape


# ——— Error on invalid dims ———
@pytest.mark.parametrize("shape, dim", SHAPE_DIM_PARAMS_INVALID)
def test_cat_invalid_dim_raises_eager(shape, dim):
    data = create_nested_dict(shape)
    td = TensorDict(data, shape)

    def cat_operation(tensor_dict_instance, cat_dimension):
        # This is the operation that is expected to raise an error
        return torch.cat(
            [tensor_dict_instance, tensor_dict_instance], dim=cat_dimension
        )

    with pytest.raises(IndexError):
        cat_operation(td, dim)


@skipif_no_compile
@pytest.mark.parametrize("shape, dim", SHAPE_DIM_PARAMS_INVALID)
def test_cat_invalid_dim_raises_compile(shape, dim):
    data = create_nested_dict(shape)
    td = TensorDict(data, shape)

    def cat_operation(tensor_dict_instance, cat_dimension):
        # This is the operation that is expected to raise an error
        return torch.cat(
            [tensor_dict_instance, tensor_dict_instance], dim=cat_dimension
        )

    # In compile mode, errors during tracing (e.g., with fake tensors by Dynamo)
    # are often wrapped in TorchRuntimeError.
    # We compile first, then expect the error upon execution of the compiled function.
    compiled_cat_op = torch.compile(cat_operation)
    with pytest.raises(IndexError, match="Dimension .* out of range"):
        compiled_cat_op(td, dim)
