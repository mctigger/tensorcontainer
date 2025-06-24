import pytest
import torch

from src.rtd.tensor_dataclass import TensorDataClass
from tests.tensor_dict.compile_utils import run_and_compare_compiled


# Helper setup
class MyTensorDataClass(TensorDataClass):
    features: torch.Tensor
    labels: torch.Tensor


def create_base_tdc(device="cpu"):
    """Creates a standard TensorDataClass instance for testing."""
    return MyTensorDataClass(
        features=torch.randn(20, 5, 10),
        labels=torch.arange(20 * 5).reshape(20, 5).float(),
        shape=(20, 5),
        device=torch.device(device),
    )


def create_source_tdc_for_slice(dest_tdc, idx):
    """Creates a source TensorDataClass with the correct shape for a given slice."""
    # The index is only used for shape calculation and not modified, so no clone is needed.
    dummy_tensor = torch.empty(dest_tdc.shape, device=dest_tdc.device)
    slice_shape = dummy_tensor[idx].shape
    return MyTensorDataClass(
        features=torch.randn(*slice_shape, 10),
        labels=torch.randn(*slice_shape),
        shape=slice_shape,
        device=dest_tdc.device,
    )


# Helper functions are now defined locally within each test
# to ensure torch.compile gets a new function object for each
# significantly different indexing scenario, avoiding recompilation limits.


# --- Test Group 1: Basic Indexing (Integers and Slices) ---


@pytest.mark.parametrize(
    "idx",
    [
        5,
        slice(2, 15),
        slice(0, 20, 3),
        (4, slice(1, 4)),
        (slice(2, 8), 3),
        (slice(1, 10, 2), slice(0, 4, 2)),
    ],
    ids=[
        "int",
        "slice",
        "slice_step",
        "int_slice_tuple",
        "slice_int_tuple",
        "slice_slice_tuple",
    ],
)
def test_setitem_basic_indexing_with_scalar(idx):
    """Tests basic indexing with a scalar is correct and compiles."""
    tdc_initial = create_base_tdc()
    value = 0.0

    def setitem_op(target_tdc, op_key, op_value):
        cloned = target_tdc.clone()
        cloned[op_key] = op_value  # op_value is scalar
        return cloned

    run_and_compare_compiled(
        setitem_op,
        tdc_initial,
        idx,
        value,
    )


@pytest.mark.parametrize(
    "idx",
    [
        5,
        slice(2, 15),
        (4, slice(1, 4)),
    ],
    ids=[
        "int_idx_tensor_val",
        "slice_idx_tensor_val",
        "tuple_idx_tensor_val",
    ],
)
def test_setitem_basic_indexing_with_broadcastable_tensor(idx):
    """Tests basic indexing with a broadcastable raw tensor value."""
    tdc_initial = create_base_tdc()
    # A tensor that is broadcastable to all fields' slices.
    # e.g. a 0-dim tensor or a tensor of shape (1,).
    value = torch.tensor(7.77, device=tdc_initial.device)

    def setitem_op(target_tdc, op_key, op_value):
        cloned = target_tdc.clone()
        cloned[op_key] = op_value  # op_value is tensor
        return cloned

    run_and_compare_compiled(
        setitem_op,
        tdc_initial,
        idx,
        value,
    )


@pytest.mark.parametrize(
    "idx", [5, slice(2, 15), (4, slice(1, 4))], ids=["int", "slice", "int_slice_tuple"]
)
def test_setitem_basic_indexing_with_tdc(idx):
    """Tests basic indexing with a TDC is correct and compiles."""
    tdc_dest_initial = create_base_tdc()
    tdc_source = create_source_tdc_for_slice(tdc_dest_initial, idx)

    def setitem_op(target_tdc, op_key, op_value):
        cloned = target_tdc.clone()
        cloned[op_key] = op_value.clone()  # op_value is TDC
        return cloned

    run_and_compare_compiled(
        setitem_op,
        tdc_dest_initial,
        idx,
        tdc_source,
    )


# --- Test Group 2: Advanced Indexing (Lists and Tensors) ---


@pytest.mark.parametrize(
    "idx",
    [
        ([0, 4, 2, 19, 7]),
        (torch.tensor([0, 4, 2, 19, 7])),
        # (torch.rand(20) > 0.5), # Moved to test_setitem_bool_mask_indexing_with_scalar
        (torch.tensor([0, 1, 2]), slice(None)),
        (slice(None), torch.tensor([0, 1, 2])),
        (torch.tensor([0, 1]), torch.tensor([2, 3])),
        # (torch.rand(20) > 0.5, slice(0, 3)), # Moved to test_setitem_bool_mask_indexing_with_scalar
    ],
    ids=[
        "list_int",
        "long_tensor",
        # "bool_mask",
        "tensor_slice_tuple",
        "slice_tensor_tuple",
        "tensor_tensor_tuple",
        # "mask_slice_tuple",
    ],
)
def test_setitem_advanced_indexing_with_scalar(idx):
    """Tests advanced indexing (excluding boolean masks) with a scalar."""
    tdc_initial = create_base_tdc()
    value = 0.0

    processed_idx = idx
    # Ensure list-of-int is converted to a tensor for compile stability
    if isinstance(idx, list) and all(isinstance(i, int) for i in idx):
        processed_idx = torch.tensor(idx, device=tdc_initial.device, dtype=torch.long)

    def setitem_op(target_tdc, op_key, op_value):
        cloned = target_tdc.clone()
        cloned[op_key] = op_value  # op_value is scalar
        return cloned

    run_and_compare_compiled(
        setitem_op,
        tdc_initial,
        processed_idx,  # Use processed_idx
        value,
    )


@pytest.mark.parametrize(
    "idx",
    [
        (torch.rand(20) > 0.5),
        (torch.rand(20) > 0.5, slice(0, 3)),
    ],
    ids=[
        "bool_mask",
        "mask_slice_tuple",
    ],
)
def test_setitem_bool_mask_indexing_with_scalar(idx):
    """Tests boolean mask indexing with a scalar using fullgraph=False."""
    tdc_initial = create_base_tdc()
    value = 0.0

    # For boolean masks, idx is already a tensor or tuple involving a tensor
    processed_idx = idx

    def setitem_op(target_tdc, op_key, op_value):
        cloned = target_tdc.clone()
        cloned[op_key] = op_value  # op_value is scalar
        return cloned

    run_and_compare_compiled(
        setitem_op,
        tdc_initial,
        processed_idx,
        value,
        fullgraph=False,  # Key change for these specific tests
    )


@pytest.mark.parametrize(
    "idx",
    [
        torch.tensor([0, 4, 2, 19, 7]),
        # torch.rand(20) > 0.5, # Moved to test_setitem_bool_mask_indexing_with_tdc
        (torch.tensor([0, 1]), torch.tensor([2, 3])),
    ],
    ids=["long_tensor", "tensor_tensor_tuple"],  # "bool_mask" removed
)
def test_setitem_advanced_indexing_with_tdc(idx):
    """Tests advanced indexing (excluding boolean masks) with a TDC."""
    tdc_dest_initial = create_base_tdc()
    tdc_source = create_source_tdc_for_slice(tdc_dest_initial, idx)

    def setitem_op(target_tdc, op_key, op_value):
        cloned = target_tdc.clone()
        cloned[op_key] = op_value.clone()  # op_value is TDC
        return cloned

    run_and_compare_compiled(
        setitem_op,
        tdc_dest_initial,
        idx,
        tdc_source,
    )


@pytest.mark.parametrize(
    "idx",
    [
        torch.rand(20) > 0.5,
    ],
    ids=["bool_mask"],
)
def test_setitem_bool_mask_indexing_with_tdc(idx):
    """Tests boolean mask indexing with a TDC using fullgraph=False."""
    tdc_dest_initial = create_base_tdc()
    tdc_source = create_source_tdc_for_slice(tdc_dest_initial, idx)

    def setitem_op(target_tdc, op_key, op_value):
        cloned = target_tdc.clone()
        cloned[op_key] = op_value.clone()  # op_value is TDC
        return cloned

    run_and_compare_compiled(
        setitem_op,
        tdc_dest_initial,
        idx,
        tdc_source,
        fullgraph=False,  # Key change for these specific tests
    )


# --- Test Group 3: Special Indexing (Ellipsis) ---


@pytest.mark.parametrize(
    "idx",
    [(Ellipsis, slice(0, 2)), (slice(0, 10), Ellipsis)],
    ids=["ellipsis_first", "ellipsis_last"],
)
def test_setitem_ellipsis_with_scalar(idx):
    """Tests ellipsis indexing with a scalar is correct and compiles."""
    tdc_initial = create_base_tdc()
    value = 0.0

    def setitem_op(target_tdc, op_key, op_value):
        cloned = target_tdc.clone()
        cloned[op_key] = op_value  # op_value is scalar
        return cloned

    run_and_compare_compiled(
        setitem_op,
        tdc_initial,
        idx,
        value,
    )


# --- Test Group 4: Edge Cases and Invalid Inputs ---


def test_setitem_broadcast_assign_tdc():
    """Tests assigning a broadcastable TDC is correct and compiles."""
    tdc_dest_initial = create_base_tdc()
    idx = slice(0, 10)
    tdc_source = MyTensorDataClass(
        features=torch.randn(1, 5, 10),
        labels=torch.randn(1, 5),
        shape=(1, 5),
        device=tdc_dest_initial.device,
    )

    def setitem_op(target_tdc, op_key, op_value):
        cloned = target_tdc.clone()
        cloned[op_key] = op_value.clone()  # op_value is TDC
        return cloned

    run_and_compare_compiled(
        setitem_op,
        tdc_dest_initial,
        idx,
        tdc_source,
    )


@pytest.mark.parametrize(
    "idx, value_shape, error, match",
    [
        (
            (slice(None),),
            (19, 5),
            ValueError,
            "The shape of the assigned TensorContainer \\(19, 5\\) is not broadcastable to the shape of the slice torch.Size\\(\\[20, 5\\]\\).",
        ),
        (20, None, IndexError, "out of bounds"),
        ((slice(None), slice(None), 0), None, IndexError, "too many indices"),
    ],
    ids=["shape_mismatch_tdc", "index_out_of_bounds", "too_many_indices"],
)
def test_setitem_invalid_inputs_raise_errors(idx, value_shape, error, match):
    """Tests that invalid inputs correctly raise errors in eager mode."""
    tdc = create_base_tdc()

    value_to_assign = 0.0  # Default to scalar for non-TDC value cases
    if value_shape is not None:  # This means value is a TDC
        value_to_assign = MyTensorDataClass(
            features=torch.randn(*value_shape, 10),
            labels=torch.randn(*value_shape),
            shape=value_shape,
            device=tdc.device,
        )

    with pytest.raises(error, match=match):
        tdc[idx] = value_to_assign


@pytest.mark.parametrize(
    "idx, tensor_value_shape, field_name_expected_to_fail, expected_slice_shape_at_fail",
    [
        (
            5,  # index
            (5, 10),  # shape of raw tensor to assign
            "labels",  # this field's slice shape won't match tensor_value_shape
            (5,),  # expected shape of tdc.labels[5]
        ),
        (
            slice(0, 2),  # index
            (2, 5, 10),  # shape of raw tensor to assign (e.g. for features field)
            "labels",  # tdc.labels[slice(0,2)] is (2,5), tensor_value_shape is (2,5,10)
            (2, 5),  # expected shape of tdc.labels[slice(0,2)]
        ),
        (
            5,  # index
            (5,),  # shape of raw tensor to assign (e.g. for labels field)
            "features",  # tdc.features[5] is (5,10), tensor_value_shape is (5,)
            (5, 10),  # expected shape of tdc.features[5]
        ),
    ],
    ids=[
        "int_idx_tensor_mismatch_on_labels",
        "slice_idx_tensor_mismatch_on_labels",
        "int_idx_tensor_mismatch_on_features",
    ],
)
def test_setitem_raw_tensor_shape_mismatch_raises_error(
    idx, tensor_value_shape, field_name_expected_to_fail, expected_slice_shape_at_fail
):
    """Tests that assigning a raw tensor with a shape that is not broadcastable
    to all field slices raises an error in eager mode."""
    tdc = create_base_tdc()
    value_tensor = torch.randn(*tensor_value_shape, device=tdc.device)

    # Constructing the expected error message part based on PyTorch's typical message
    # and how TensorDataClass wraps it.
    # Example PyTorch error: "shape mismatch: value tensor of shape torch.Size([5, 10]) cannot be broadcast to indexing result of shape torch.Size([5])"
    # TensorDataClass wraps it as: "Error assigning to field '{field_key}': {original_error_message}"
    original_pytorch_error_detail = (
        f"shape mismatch: value tensor of shape torch.Size({list(tensor_value_shape)}) "
        f"cannot be broadcast to indexing result of shape torch.Size({list(expected_slice_shape_at_fail)})"
    )
    expected_match_pattern = f"Error assigning to field '{field_name_expected_to_fail}': {original_pytorch_error_detail}"
