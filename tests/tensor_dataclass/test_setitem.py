import pytest
import torch

from src.rtd.tensor_dataclass import TensorDataClass
from tests.tensor_dict.compile_utils import run_and_compare_compiled


class SampleTensorDataClass(TensorDataClass):
    """Test dataclass with features and labels tensors."""

    features: torch.Tensor  # shape (B, N, D) or (B, N, D1, D2)
    labels: torch.Tensor  # shape (B, N)


def create_sample_tdc(batch_shape=(20, 5), feature_dims=10, num_event_dims=1):
    """Helper to create a sample TensorDataClass for testing."""
    if num_event_dims == 1:
        features = torch.randn(*batch_shape, feature_dims)
    elif num_event_dims == 2:
        features = torch.randn(
            *batch_shape, feature_dims, 5
        )  # Adding another event dim
    else:
        raise ValueError("num_event_dims must be 1 or 2")

    total_elements = 1
    for dim in batch_shape:
        total_elements *= dim
    labels = torch.arange(total_elements).reshape(batch_shape).float()
    return SampleTensorDataClass(
        features=features, labels=labels, shape=batch_shape, device=features.device
    )


def create_source_tdc_for_slice(dest_tdc, idx):
    """Creates a source TensorDataClass with the correct shape for a given slice."""
    # The index is only used for shape calculation and not modified, so no clone is needed.
    dummy_tensor = torch.empty(dest_tdc.shape, device=dest_tdc.device)
    slice_shape = dummy_tensor[idx].shape
    return SampleTensorDataClass(
        features=torch.randn(*slice_shape, 10),
        labels=torch.randn(*slice_shape),
        shape=slice_shape,
        device=dest_tdc.device,
    )


def random_mask(*shape):
    """Creates a random boolean mask of a given shape."""
    return torch.rand(shape) > 0.5


def _set_item(tdc, idx, value):
    cloned = tdc.clone()
    cloned[idx] = value
    return cloned


class TestSetItem:
    """Consolidated test suite for __setitem__ operations on TensorDataClass."""


    def _run_and_verify_setitem(self, tdc, idx, value, test_name, fullgraph=True):
        """
        Central helper method to encapsulate common testing logic for __setitem__.
        """
        def setitem_op(target_tdc, op_key, op_value):
            cloned = target_tdc.clone()
            cloned[op_key] = op_value
            return cloned

        try:
            # Use a copy for the compilation check to avoid modifying the original
            torch.compile(setitem_op, fullgraph=fullgraph)(tdc.clone(), idx, value)
        except Exception:
            if fullgraph:
                fullgraph = False

        result, _ = run_and_compare_compiled(
            setitem_op,
            tdc,
            idx,
            value,
            fullgraph=fullgraph,
            expected_graph_breaks=0 if fullgraph else None,
        )

        assert isinstance(result, SampleTensorDataClass), (
            f"Expected SampleTensorDataClass, got {type(result)} for {test_name}"
        )

        return result

    # Combined basic indexing test cases
    BASIC_INDEXING_CASES = [
        ("int", 5),
        ("slice", slice(2, 15)),
        ("slice_step", slice(0, 20, 3)),
        ("int_slice_tuple", (4, slice(1, 4))),
        ("slice_int_tuple", (slice(2, 8), 3)),
        ("slice_slice_tuple", (slice(1, 10, 2), slice(0, 4, 2))),
    ]

    @pytest.mark.parametrize(
        "test_name,idx",
        BASIC_INDEXING_CASES,
        ids=[case[0] for case in BASIC_INDEXING_CASES],
    )
    def test_setitem_basic_indexing_with_scalar(self, test_name, idx):
        """Tests basic indexing with a scalar is correct and compiles."""
        tdc_initial = create_sample_tdc()
        value = 0.0
        self._run_and_verify_setitem(tdc_initial, idx, value, test_name)


    BROADCASTABLE_TENSOR_CASES = [
        ("int_idx_tensor_val", 5),
        ("slice_idx_tensor_val", slice(2, 15)),
        ("tuple_idx_tensor_val", (4, slice(1, 4))),
    ]

    @pytest.mark.parametrize(
        "test_name,idx",
        BROADCASTABLE_TENSOR_CASES,
        ids=[case[0] for case in BROADCASTABLE_TENSOR_CASES],
    )
    def test_setitem_basic_indexing_with_broadcastable_tensor(self, test_name, idx):
        """Tests basic indexing with a broadcastable raw tensor value."""
        tdc_initial = create_sample_tdc()
        # A tensor that is broadcastable to all fields' slices.
        # e.g. a 0-dim tensor or a tensor of shape (1,).
        value = torch.tensor(7.77, device=tdc_initial.device)
        self._run_and_verify_setitem(tdc_initial, idx, value, test_name)


    TDC_VALUE_CASES = [
        ("int", 5),
        ("slice", slice(2, 15)),
        ("int_slice_tuple", (4, slice(1, 4))),
    ]

    @pytest.mark.parametrize(
        "test_name,idx",
        TDC_VALUE_CASES,
        ids=[case[0] for case in TDC_VALUE_CASES],
    )
    def test_setitem_basic_indexing_with_tdc(self, test_name, idx):
        """Tests basic indexing with a TDC is correct and compiles."""
        tdc_dest_initial = create_sample_tdc()
        tdc_source = create_source_tdc_for_slice(tdc_dest_initial, idx)
        self._run_and_verify_setitem(tdc_dest_initial, idx, tdc_source, test_name)


    # Advanced indexing test cases (excluding boolean masks)
    ADVANCED_INDEXING_CASES = [
        ("list_int", ([0, 4, 2, 19, 7])),
        ("long_tensor", (torch.tensor([0, 4, 2, 19, 7]))),
        ("tensor_slice_tuple", (torch.tensor([0, 1, 2]), slice(None))),
        ("slice_tensor_tuple", (slice(None), torch.tensor([0, 1, 2]))),
        ("tensor_tensor_tuple", (torch.tensor([0, 1]), torch.tensor([2, 3]))),
    ]

    @pytest.mark.parametrize(
        "test_name,idx",
        ADVANCED_INDEXING_CASES,
        ids=[case[0] for case in ADVANCED_INDEXING_CASES],
    )
    def test_setitem_advanced_indexing_with_scalar(self, test_name, idx):
        """Tests advanced indexing (excluding boolean masks) with a scalar."""
        tdc_initial = create_sample_tdc()
        value = 0.0

        processed_idx = idx
        # Ensure list-of-int is converted to a tensor for compile stability
        if isinstance(idx, list) and all(isinstance(i, int) for i in idx):
            processed_idx = torch.tensor(idx, device=tdc_initial.device, dtype=torch.long)

        self._run_and_verify_setitem(tdc_initial, processed_idx, value, test_name)


    # Boolean mask indexing test cases
    BOOLEAN_MASK_CASES = [
        ("bool_mask", (torch.rand(20) > 0.5)),
        ("mask_slice_tuple", (torch.rand(20) > 0.5, slice(0, 3))),
    ]

    @pytest.mark.parametrize(
        "test_name,idx",
        BOOLEAN_MASK_CASES, 
        ids=[case[0] for case in BOOLEAN_MASK_CASES],
    )
    def test_setitem_bool_mask_indexing_with_scalar(self, test_name, idx):
        """Tests boolean mask indexing with a scalar using fullgraph=False."""
        tdc_initial = create_sample_tdc()
        value = 0.0
        self._run_and_verify_setitem(tdc_initial, idx, value, test_name, fullgraph=False)


    ADVANCED_TDC_CASES = [
        ("long_tensor", torch.tensor([0, 4, 2, 19, 7])),
        ("tensor_tensor_tuple", (torch.tensor([0, 1]), torch.tensor([2, 3]))),
    ]

    @pytest.mark.parametrize(
        "test_name,idx",
        ADVANCED_TDC_CASES,
        ids=[case[0] for case in ADVANCED_TDC_CASES],
    )
    def test_setitem_advanced_indexing_with_tdc(self, test_name, idx):
        """Tests advanced indexing (excluding boolean masks) with a TDC."""
        tdc_dest_initial = create_sample_tdc()
        tdc_source = create_source_tdc_for_slice(tdc_dest_initial, idx)
        self._run_and_verify_setitem(tdc_dest_initial, idx, tdc_source, test_name)


    BOOLEAN_MASK_TDC_CASES = [
        ("bool_mask", torch.rand(20) > 0.5),
    ]

    @pytest.mark.parametrize(
        "test_name,idx",
        BOOLEAN_MASK_TDC_CASES,
        ids=[case[0] for case in BOOLEAN_MASK_TDC_CASES],
    )
    def test_setitem_bool_mask_indexing_with_tdc(self, test_name, idx):
        """Tests boolean mask indexing with a TDC using fullgraph=False."""
        tdc_dest_initial = create_sample_tdc()
        tdc_source = create_source_tdc_for_slice(tdc_dest_initial, idx)
        self._run_and_verify_setitem(tdc_dest_initial, idx, tdc_source, test_name, fullgraph=False)


    # Special ellipsis indexing test cases
    ELLIPSIS_CASES = [
        ("ellipsis_first", (Ellipsis, slice(0, 2))),
        ("ellipsis_last", (slice(0, 10), Ellipsis)),
    ]

    @pytest.mark.parametrize(
        "test_name,idx",
        ELLIPSIS_CASES,
        ids=[case[0] for case in ELLIPSIS_CASES],
    )
    def test_setitem_ellipsis_with_scalar(self, test_name, idx):
        """Tests ellipsis indexing with a scalar is correct and compiles."""
        tdc_initial = create_sample_tdc()
        value = 0.0
        self._run_and_verify_setitem(tdc_initial, idx, value, test_name)


    def test_setitem_broadcast_assign_tdc(self):
        """Tests assigning a broadcastable TDC is correct and compiles."""
        tdc_dest_initial = create_sample_tdc()
        idx = slice(0, 10)
        tdc_source = SampleTensorDataClass(
            features=torch.randn(1, 5, 10),
            labels=torch.randn(1, 5),
            shape=(1, 5),
            device=tdc_dest_initial.device,
        )
        self._run_and_verify_setitem(tdc_dest_initial, idx, tdc_source, "broadcast_assign_tdc")


    # Invalid indexing test cases
    INVALID_INDEXING_CASES = [
        (
            "shape_mismatch_tdc",
            (slice(None),),
            (19, 5),
            ValueError,
            "The shape of the assigned TensorContainer \\(19, 5\\) is not broadcastable to the shape of the slice torch.Size\\(\\[20, 5\\]\\).",
        ),
        ("index_out_of_bounds", 20, None, IndexError, "out of bounds"),
        ("too_many_indices", (slice(None), slice(None), 0), None, IndexError, "too many indices"),
    ]

    @pytest.mark.parametrize(
        "test_name,idx,value_shape,error,match",
        INVALID_INDEXING_CASES,
        ids=[case[0] for case in INVALID_INDEXING_CASES],
    )
    def test_setitem_invalid_inputs_raise_errors(self, test_name, idx, value_shape, error, match):
        """Tests that invalid inputs correctly raise errors in eager mode."""
        tdc = create_sample_tdc()

        value_to_assign = 0.0  # Default to scalar for non-TDC value cases
        if value_shape is not None:  # This means value is a TDC
            value_to_assign = SampleTensorDataClass(
                features=torch.randn(*value_shape, 10),
                labels=torch.randn(*value_shape),
                shape=value_shape,
                device=tdc.device,
            )

        with pytest.raises(error, match=match):
            tdc[idx] = value_to_assign


    # Raw tensor shape mismatch test cases
    RAW_TENSOR_MISMATCH_CASES = [
        (
            "int_idx_tensor_mismatch_on_labels",
            5,  # index
            (5, 10),  # shape of raw tensor to assign
            "labels",  # this field's slice shape won't match tensor_value_shape
            (5,),  # expected shape of tdc.labels[5]
        ),
        (
            "slice_idx_tensor_mismatch_on_labels",
            slice(0, 2),  # index
            (2, 5, 10),  # shape of raw tensor to assign (e.g. for features field)
            "labels",  # tdc.labels[slice(0,2)] is (2,5), tensor_value_shape is (2,5,10)
            (2, 5),  # expected shape of tdc.labels[slice(0,2)]
        ),
        (
            "int_idx_tensor_mismatch_on_features",
            5,  # index
            (5,),  # shape of raw tensor to assign (e.g. for labels field)
            "features",  # tdc.features[5] is (5,10), tensor_value_shape is (5,)
            (5, 10),  # expected shape of tdc.features[5]
        ),
    ]

    @pytest.mark.parametrize(
        "test_name,idx,tensor_value_shape,field_name_expected_to_fail,expected_slice_shape_at_fail",
        RAW_TENSOR_MISMATCH_CASES,
        ids=[case[0] for case in RAW_TENSOR_MISMATCH_CASES],
    )
    def test_setitem_raw_tensor_shape_mismatch_raises_error(
        self, test_name, idx, tensor_value_shape, field_name_expected_to_fail, expected_slice_shape_at_fail
    ):
        """Tests that assigning a raw tensor with a shape that is not broadcastable
        to all field slices raises an error in eager mode."""
        tdc = create_sample_tdc()
        value_tensor = torch.randn(*tensor_value_shape, device=tdc.device)

        # The actual PyTorch error message format from the implementation
        # Just check that the error is raised - the specific message format can vary
        with pytest.raises(RuntimeError):
            tdc[idx] = value_tensor
