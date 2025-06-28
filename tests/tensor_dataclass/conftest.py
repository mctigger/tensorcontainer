import dataclasses
from typing import List, Optional

import pytest
import torch

from rtd.tensor_dataclass import TensorDataClass

# ============================================================================
# COMMON TENSORDATACLASS DEFINITIONS
# ============================================================================


# Define a simple, reusable TensorDataclass for tests
class SimpleTensorData(TensorDataClass):
    a: torch.Tensor
    b: torch.Tensor


class DeviceTestClass(TensorDataClass):
    """TensorDataclass for device-related testing."""

    device: Optional[torch.device]
    shape: tuple
    a: torch.Tensor
    b: torch.Tensor
    meta: int = 42


class CloneTestClass(TensorDataClass):
    """TensorDataclass for clone-related testing."""

    a: torch.Tensor
    b: torch.Tensor
    meta: int = 42


class StackTestClass(TensorDataClass):
    """TensorDataclass for stack-related testing."""

    shape: tuple
    device: Optional[torch.device]
    a: torch.Tensor
    b: torch.Tensor
    meta: int = 42


class ToTestClass(TensorDataClass):
    """TensorDataclass for to() method testing."""

    device: Optional[torch.device]
    shape: tuple
    a: torch.Tensor
    b: torch.Tensor
    meta: int = 42


class ShapeTestClass(TensorDataClass):
    """TensorDataclass for shape-related testing."""

    shape: tuple
    device: Optional[torch.device]
    a: torch.Tensor
    b: torch.Tensor
    meta: int = 42


class OptionalFieldsTestClass(TensorDataClass):
    """TensorDataclass with optional and default_factory fields for testing."""

    obs: torch.Tensor
    reward: Optional[torch.Tensor]
    info: List[str] = dataclasses.field(default_factory=list)
    optional_meta: Optional[str] = None
    optional_meta_val: Optional[str] = "value"
    default_tensor: torch.Tensor = dataclasses.field(
        default_factory=lambda: torch.zeros(4)
    )


class ViewReshapeTestClass(TensorDataClass):
    """TensorDataclass for view/reshape method testing."""

    shape: tuple
    device: Optional[torch.device]
    a: torch.Tensor
    b: torch.Tensor


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def simple_tensor_data_instance():
    """Returns a SimpleTensorData instance with random tensors of shape (3, 4)."""
    return SimpleTensorData(
        a=torch.randn(3, 4),
        b=torch.randn(3, 4),
        shape=(3, 4),
        device=torch.device("cpu"),
    )


@pytest.fixture
def device_test_instance():
    """Returns a DeviceTestClass instance on CPU."""
    return DeviceTestClass(
        a=torch.randn(2, 3),
        b=torch.ones(2, 3),
        shape=(2, 3),
        device=torch.device("cpu"),
    )


@pytest.fixture
def cuda_device_test_instance():
    """Returns a DeviceTestClass instance on CUDA if available, otherwise skips."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return DeviceTestClass(
        a=torch.randn(2, 3, device="cuda"),
        b=torch.ones(2, 3, device="cuda"),
        shape=(2, 3),
        device=torch.device("cuda"),
    )


@pytest.fixture
def clone_test_instance():
    """Returns a CloneTestClass instance with requires_grad=True tensors."""
    return CloneTestClass(
        a=torch.randn(2, 3, requires_grad=True),
        b=torch.ones(2, 3, requires_grad=True),
        shape=(2, 3),
        device=torch.device("cpu"),
    )


@pytest.fixture
def stack_test_instances():
    """Returns a pair of StackTestClass instances for stacking tests."""
    td1 = StackTestClass(
        a=torch.randn(2, 3),
        b=torch.ones(2, 3),
        shape=(2, 3),
        device=torch.device("cpu"),
    )
    td2 = StackTestClass(
        a=torch.randn(2, 3),
        b=torch.ones(2, 3),
        shape=(2, 3),
        device=torch.device("cpu"),
    )
    return td1, td2


@pytest.fixture
def shape_test_instance():
    """Returns a ShapeTestClass instance for shape operations."""
    return ShapeTestClass(
        shape=(4, 5),
        device=torch.device("cpu"),
        a=torch.randn(4, 5),
        b=torch.ones(4, 5),
    )


@pytest.fixture
def optional_fields_instance():
    """Returns an OptionalFieldsTestClass instance with None reward."""
    return OptionalFieldsTestClass(
        shape=(4,),
        device=None,
        obs=torch.randn(4, 32, 32),
        reward=None,
        info=["step1"],
    )


@pytest.fixture
def to_test_instance():
    """Returns a ToTestClass instance for .to() method testing."""
    return ToTestClass(
        a=torch.randn(2, 3, device=torch.device("cpu")),
        b=torch.ones(2, 3, device=torch.device("cpu")),
        shape=(2, 3),
        device=torch.device("cpu"),
    )


@pytest.fixture
def to_test_4d_instance():
    """Returns a ToTestClass instance with 4D tensors for memory format testing."""
    return ToTestClass(
        a=torch.randn(2, 3, 4, 5, device=torch.device("cpu")),
        b=torch.ones(2, 3, 4, 5, device=torch.device("cpu")),
        shape=(2, 3, 4, 5),
        device=torch.device("cpu"),
    )


@pytest.fixture
def view_reshape_test_instance():
    """Returns a ViewReshapeTestClass instance for view/reshape testing."""
    return ViewReshapeTestClass(
        a=torch.randn(4, 5),
        b=torch.ones(4, 5),
        shape=(4, 5),
        device=torch.device("cpu"),
    )


@pytest.fixture
def view_reshape_test_instance_2x6():
    """Returns a ViewReshapeTestClass instance with 2x6 shape for reshape testing."""
    return ViewReshapeTestClass(
        a=torch.randn(2, 6),
        b=torch.ones(2, 6),
        shape=(2, 6),
        device=torch.device("cpu"),
    )


@pytest.fixture
def view_reshape_test_instance_5x4():
    """Returns a ViewReshapeTestClass instance with 5x4 shape for transpose testing."""
    return ViewReshapeTestClass(
        a=torch.randn(5, 4),
        b=torch.randn(5, 4),
        shape=(5, 4),
        device=torch.device("cpu"),
    )


# ============================================================================
# PARAMETER SETS FOR TESTING
# ============================================================================

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

# Common shapes for testing
COMMON_SHAPES = [
    (4,),
    (2, 3),
    (1, 4),
    (2, 1, 2),
    (3, 4, 5),
    (0, 10),  # Zero-sized batch
]

# Common devices for testing
COMMON_DEVICES = [
    torch.device("cpu"),
    pytest.param(
        torch.device("cuda"),
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA not available"
        ),
    ),
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def compute_cat_shape(shape, dim):
    """Helper function to compute expected shape after cat."""
    ndim = len(shape)
    pos_dim = dim if dim >= 0 else dim + ndim
    expected_shape = list(shape)
    expected_shape[pos_dim] = expected_shape[pos_dim] * 2
    return tuple(expected_shape)


def compute_stack_shape(shape, dim):
    """Helper function to compute expected shape after stack."""
    ndim = len(shape)
    pos_dim = dim if dim >= 0 else dim + ndim + 1
    expected_shape = list(shape)
    expected_shape.insert(pos_dim, 2)  # Assuming stacking 2 tensors
    return tuple(expected_shape)


def create_tensor_with_device(shape, device, dtype=None, requires_grad=False):
    """Helper function to create tensors with specified device and properties."""
    if dtype is None:
        dtype = torch.float32
    return torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)


def create_ones_tensor_with_device(shape, device, dtype=None, requires_grad=False):
    """Helper function to create ones tensors with specified device and properties."""
    if dtype is None:
        dtype = torch.float32
    return torch.ones(shape, device=device, dtype=dtype, requires_grad=requires_grad)


def create_zeros_tensor_with_device(shape, device, dtype=None, requires_grad=False):
    """Helper function to create zeros tensors with specified device and properties."""
    if dtype is None:
        dtype = torch.float32
    return torch.zeros(shape, device=device, dtype=dtype, requires_grad=requires_grad)


def assert_tensor_equal_and_different_objects(tensor1, tensor2):
    """Helper to assert tensors are equal in value but different objects."""
    assert torch.equal(tensor1, tensor2)
    assert tensor1 is not tensor2


def assert_device_consistency(tensor_dataclass_instance, expected_device):
    """Helper to assert device consistency across TensorDataclass and its tensors."""
    if tensor_dataclass_instance.device is not None:
        assert tensor_dataclass_instance.device.type == expected_device.type

    # Check all tensor fields
    for field_name, field_value in tensor_dataclass_instance.__dict__.items():
        if isinstance(field_value, torch.Tensor):
            assert field_value.device.type == expected_device.type


def assert_shape_consistency(tensor_dataclass_instance, expected_shape):
    """Helper to assert shape consistency across TensorDataclass and its tensors."""
    assert tensor_dataclass_instance.shape == expected_shape

    # Check all tensor fields have compatible shapes
    for field_name, field_value in tensor_dataclass_instance.__dict__.items():
        if isinstance(field_value, torch.Tensor):
            # Check that tensor shape starts with the expected batch shape
            tensor_shape = field_value.shape
            batch_dims = len(expected_shape)
            assert tensor_shape[:batch_dims] == expected_shape


def create_inconsistent_meta_data_pair(base_class, shape, device, meta_values):
    """Helper to create a pair of TensorDataclass instances with inconsistent metadata."""
    td1 = base_class(
        a=torch.randn(shape),
        b=torch.ones(shape),
        shape=shape,
        device=device,
        meta=meta_values[0],
    )
    td2 = base_class(
        a=torch.randn(shape),
        b=torch.ones(shape),
        shape=shape,
        device=device,
        meta=meta_values[1],
    )
    return td1, td2


def create_inconsistent_shape_pair(base_class, shapes, device):
    """Helper to create a pair of TensorDataclass instances with inconsistent shapes."""
    td1 = base_class(
        a=torch.randn(shapes[0]),
        b=torch.ones(shapes[0]),
        shape=shapes[0],
        device=device,
    )
    td2 = base_class(
        a=torch.randn(shapes[1]),
        b=torch.ones(shapes[1]),
        shape=shapes[1],
        device=device,
    )
    return td1, td2


def create_nested_tensor_dataclass():
    """Helper to create nested TensorDataclass instances for testing."""

    class Inner(TensorDataClass):
        shape: tuple
        device: Optional[torch.device]
        c: torch.Tensor

    class Outer(TensorDataClass):
        shape: tuple
        device: Optional[torch.device]
        inner: Inner

    inner = Inner(
        c=torch.randn(2, 3, requires_grad=True),
        shape=(2, 3),
        device=torch.device("cpu"),
    )

    outer = Outer(
        inner=inner,
        shape=(2, 3),
        device=torch.device("cpu"),
    )

    return outer, inner


@pytest.fixture
def nested_tensor_data_instance():
    """Returns a nested TensorDataclass instance."""
    outer, _ = create_nested_tensor_dataclass()
    return outer


class CopyTestClass(TensorDataClass):
    a: torch.Tensor
    b: torch.Tensor
    metadata: list


@pytest.fixture
def copy_test_instance():
    """Returns a CopyTestClass instance with a list as metadata."""
    return CopyTestClass(
        a=torch.randn(3, 4),
        b=torch.randn(3, 4),
        metadata=[1, 2, 3],
        shape=(3, 4),
        device=torch.device("cpu"),
    )


# ============================================================================
# COMPLEX TENSORDATACLASS DEFINITIONS FOR REFACTORING
# ============================================================================


class FlatTensorDataClass(TensorDataClass):
    tensor: torch.Tensor
    meta_data: str


class NestedTensorDataClass(TensorDataClass):
    tensor: torch.Tensor
    tensor_data_class: FlatTensorDataClass
    meta_data: str
    optional_tensor: Optional[torch.Tensor] = None
    optional_meta_data: Optional[str] = None


# ============================================================================
# REFACTORING FIXTURES
# ============================================================================


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ]
)
def nested_tensor_data_class(request):
    """
    Returns a complex, nested TensorDataclass instance on the specified device.
    """
    device = torch.device(request.param)
    batch_shape = (2, 3)
    event_shape = (4, 5)

    # Create the nested dataclass instance
    flat = FlatTensorDataClass(
        tensor=torch.randn(*batch_shape, *event_shape, device=device),
        meta_data="meta_data_str",
        shape=batch_shape,
        device=device,
    )

    # Create the main dataclass instance
    tdc = NestedTensorDataClass(
        tensor=torch.randn(*batch_shape, *event_shape, device=device),
        tensor_data_class=flat,
        meta_data="meta_data_str",
        shape=batch_shape,
        device=device,
    )

    return tdc


# ============================================================================
# ASSERTION HELPERS
# ============================================================================


def assert_compilation_works(func, *args, **kwargs):
    """Helper to assert that a function can be compiled and produces consistent results."""
    from tests.compile_utils import run_and_compare_compiled

    run_and_compare_compiled(func, *args, **kwargs)


def assert_raises_with_message(
    exception_type, message_pattern, callable_obj, *args, **kwargs
):
    """Helper to assert that a callable raises a specific exception with a message pattern."""
    with pytest.raises(exception_type, match=message_pattern):
        callable_obj(*args, **kwargs)


def assert_tensor_properties(
    tensor,
    expected_shape=None,
    expected_device=None,
    expected_dtype=None,
    expected_requires_grad=None,
):
    """Helper to assert multiple tensor properties at once."""
    if expected_shape is not None:
        assert tensor.shape == expected_shape
    if expected_device is not None:
        assert tensor.device.type == expected_device.type
    if expected_dtype is not None:
        assert tensor.dtype == expected_dtype
    if expected_requires_grad is not None:
        assert tensor.requires_grad == expected_requires_grad
