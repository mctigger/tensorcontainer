import pytest
import torch
from torch._dynamo import exc as dynamo_exc

from rtd.tensor_dataclass import TensorDataclass
from tests.conftest import skipif_no_compile
from tests.tensor_dict.compile_utils import run_and_compare_compiled


# Define a base class for tests
class CatTestClass(TensorDataclass):
    a: torch.Tensor
    b: torch.Tensor
    meta: int = 42


# Define parameter sets for valid and invalid dimensions
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


# Helper function to compute expected shape after cat
def compute_cat_shape(shape, dim):
    ndim = len(shape)
    pos_dim = dim if dim >= 0 else dim + ndim
    expected_shape = list(shape)
    expected_shape[pos_dim] = expected_shape[pos_dim] * 2
    return tuple(expected_shape)


# --- Valid concatenation dims across several shapes ---
@pytest.mark.parametrize("shape, dim", SHAPE_DIM_PARAMS_VALID)
def test_cat_valid(shape, dim):
    td1 = CatTestClass(
        a=torch.randn(shape),
        b=torch.ones(shape),
        shape=shape,
        device=torch.device("cpu"),
    )
    td2 = CatTestClass(
        a=torch.randn(shape),
        b=torch.ones(shape),
        shape=shape,
        device=torch.device("cpu"),
    )

    def cat_operation(tensor_dict_instance_list, dim_arg):
        return torch.cat(tensor_dict_instance_list, dim=dim_arg)

    # Skip compilation for all cases
    cat_td: CatTestClass = cat_operation([td1, td2], dim)

    expected_shape = compute_cat_shape(shape, dim)
    assert cat_td.shape == expected_shape
    assert cat_td.a.shape == expected_shape
    assert cat_td.b.shape == expected_shape
    assert cat_td.meta == 42

    # Replace compare_nested_dict with direct tensor comparisons
    if dim < 0:
        actual_dim = dim + len(shape)
    else:
        actual_dim = dim
    split_point = shape[actual_dim]

    slices_td1_list = [slice(None)] * len(shape)
    slices_td2_list = [slice(None)] * len(shape)
    slices_td1_list[actual_dim] = slice(0, split_point)
    slices_td2_list[actual_dim] = slice(split_point, None)

    slices_td1 = tuple(slices_td1_list)
    slices_td2 = tuple(slices_td2_list)

    assert torch.equal(cat_td.a[slices_td1], td1.a)
    assert torch.equal(cat_td.a[slices_td2], td2.a)
    assert torch.equal(cat_td.b[slices_td1], td1.b)
    assert torch.equal(cat_td.b[slices_td2], td2.b)


# --- Error on invalid dims ---
@pytest.mark.parametrize("shape, dim", SHAPE_DIM_PARAMS_INVALID)
def test_cat_invalid_dim_raises_eager(shape, dim):
    td1 = CatTestClass(
        a=torch.randn(shape),
        b=torch.ones(shape),
        shape=shape,
        device=torch.device("cpu"),
    )
    td2 = CatTestClass(
        a=torch.randn(shape),
        b=torch.ones(shape),
        shape=shape,
        device=torch.device("cpu"),
    )

    def cat_operation(tensor_dict_instance_list, dim_arg):
        return torch.cat(tensor_dict_instance_list, dim=dim_arg)

    with pytest.raises(IndexError):
        cat_operation([td1, td2], dim)


@skipif_no_compile
@pytest.mark.parametrize("shape, dim", SHAPE_DIM_PARAMS_INVALID)
def test_cat_invalid_dim_raises_compile(shape, dim):
    td1 = CatTestClass(
        a=torch.randn(shape),
        b=torch.ones(shape),
        shape=shape,
        device=torch.device("cpu"),
    )
    td2 = CatTestClass(
        a=torch.randn(shape),
        b=torch.ones(shape),
        shape=shape,
        device=torch.device("cpu"),
    )

    def cat_operation(tensor_dict_instance_list, dim_arg):
        return torch.cat(tensor_dict_instance_list, dim=dim_arg)

    compiled_cat_op = torch.compile(cat_operation, fullgraph=True)
    with pytest.raises(dynamo_exc.TorchRuntimeError) as excinfo:
        compiled_cat_op([td1, td2], dim)
    assert "IndexError" in str(excinfo.value) and "Dimension out of range" in str(
        excinfo.value
    )


# --- Test cat on CUDA ---
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cat_on_cuda_eager():
    td1 = CatTestClass(
        a=torch.randn(2, 3, device="cuda"),
        b=torch.ones(2, 3, device="cuda"),
        shape=(2, 3),
        device=torch.device("cuda"),
    )
    td2 = CatTestClass(
        a=torch.randn(2, 3, device="cuda"),
        b=torch.ones(2, 3, device="cuda"),
        shape=(2, 3),
        device=torch.device("cuda"),
    )

    def cat_operation_cuda(tensor_dict_instance_list, dim_arg):
        return torch.cat(tensor_dict_instance_list, dim=dim_arg)

    # Test eager mode
    cat_td_eager: CatTestClass = cat_operation_cuda([td1, td2], 0)
    assert cat_td_eager.device.type == "cuda"
    assert cat_td_eager.a.device.type == "cuda"
    assert cat_td_eager.b.device.type == "cuda"
    assert cat_td_eager.shape == (4, 3)


@skipif_no_compile
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cat_on_cuda_compile():
    td1 = CatTestClass(
        a=torch.randn(2, 3, device="cuda"),
        b=torch.ones(2, 3, device="cuda"),
        shape=(2, 3),
        device=torch.device("cuda"),
    )
    td2 = CatTestClass(
        a=torch.randn(2, 3, device="cuda"),
        b=torch.ones(2, 3, device="cuda"),
        shape=(2, 3),
        device=torch.device("cuda"),
    )

    def cat_operation_cuda(tensor_dict_instance_list, dim_arg):
        return torch.cat(tensor_dict_instance_list, dim=dim_arg)

    # Test compile mode
    run_and_compare_compiled(cat_operation_cuda, [td1, td2], 0)


# --- Test cat with inconsistent meta data raises ---
def test_cat_inconsistent_meta_data_raises():
    td1 = CatTestClass(
        a=torch.randn(2, 3),
        b=torch.ones(2, 3),
        shape=(2, 3),
        device=torch.device("cpu"),
        meta=42,
    )
    td2 = CatTestClass(
        a=torch.randn(2, 3),
        b=torch.ones(2, 3),
        shape=(2, 3),
        device=torch.device("cpu"),
        meta=99,  # Inconsistent meta data
    )

    def cat_operation(tensor_dict_instance_list, dim_arg):
        return torch.cat(tensor_dict_instance_list, dim=dim_arg)

    with pytest.raises(ValueError):
        cat_operation([td1, td2], 0)


@skipif_no_compile
def test_cat_inconsistent_meta_data_raises_compile():
    td1 = CatTestClass(
        a=torch.randn(2, 3),
        b=torch.ones(2, 3),
        shape=(2, 3),
        device=torch.device("cpu"),
        meta=42,
    )
    td2 = CatTestClass(
        a=torch.randn(2, 3),
        b=torch.ones(2, 3),
        shape=(2, 3),
        device=torch.device("cpu"),
        meta=99,  # Inconsistent meta data
    )

    def cat_operation(tensor_dict_instance_list, dim_arg):
        return torch.cat(tensor_dict_instance_list, dim=dim_arg)

    # Don't compile the function that raises an exception
    with pytest.raises(dynamo_exc.Unsupported):
        torch.compile(cat_operation, fullgraph=True)([td1, td2], 0)


# --- Test cat with inconsistent shapes raises ---
def test_cat_inconsistent_shapes_raises():
    td1 = CatTestClass(
        a=torch.randn(2, 3),
        b=torch.ones(2, 3),
        shape=(2, 3),
        device=torch.device("cpu"),
    )
    td2 = CatTestClass(
        a=torch.randn(2, 4),  # Inconsistent shape
        b=torch.ones(2, 4),
        shape=(2, 4),
        device=torch.device("cpu"),
    )

    def cat_operation(tensor_dict_instance_list, dim_arg):
        return torch.cat(tensor_dict_instance_list, dim=dim_arg)

    with pytest.raises(ValueError):
        cat_operation([td1, td2], 0)


@skipif_no_compile
def test_cat_inconsistent_shapes_raises_compile():
    td1 = CatTestClass(
        a=torch.randn(2, 3),
        b=torch.ones(2, 3),
        shape=(2, 3),
        device=torch.device("cpu"),
    )
    td2 = CatTestClass(
        a=torch.randn(2, 4),  # Inconsistent shape
        b=torch.ones(2, 4),
        shape=(2, 4),
        device=torch.device("cpu"),
    )

    def cat_operation(tensor_dict_instance_list, dim_arg):
        return torch.cat(tensor_dict_instance_list, dim=dim_arg)

    # Don't compile the function that raises an exception
    with pytest.raises(dynamo_exc.Unsupported):
        torch.compile(cat_operation, fullgraph=True)([td1, td2], 0)


# --- Test compile functionality for torch.cat ---
@skipif_no_compile
def test_cat_compile():
    """Tests that a function using torch.cat with TensorDataclass can be torch.compiled."""

    class MyData(TensorDataclass):
        x: torch.Tensor
        y: torch.Tensor

    def func(tds: list[MyData], dim_arg: int) -> MyData:
        return torch.cat(tds, dim=dim_arg)

    data1 = MyData(
        x=torch.ones(3, 4),
        y=torch.zeros(3, 4),
        shape=(3, 4),
        device=torch.device("cpu"),
    )

    data2 = MyData(
        x=torch.ones(3, 4) * 2,
        y=torch.zeros(3, 4) * 2,
        shape=(3, 4),
        device=torch.device("cpu"),
    )
    run_and_compare_compiled(func, [data1, data2], 0)


def test_cat_empty_list_raises():
    """Test that torch.cat on an empty list raises a RuntimeError."""
    with pytest.raises(RuntimeError, match="expected a non-empty list of Tensors"):
        torch.cat([], dim=0)


def test_cat_inconsistent_non_cat_dim_raises():
    """
    Test that torch.cat raises a ValueError for inconsistent non-cat dimensions.
    """
    td1 = CatTestClass(
        shape=(2, 3),
        device=torch.device("cpu"),
        a=torch.randn(2, 3),
        b=torch.randn(2, 3),
    )
    td2 = CatTestClass(
        shape=(2, 4),  # Inconsistent dimension 1
        device=torch.device("cpu"),
        a=torch.randn(2, 4),
        b=torch.randn(2, 4),
    )

    with pytest.raises(ValueError, match="Node context mismatch"):
        torch.cat([td1, td2], dim=0)  # Concatenating along dimension 0
