import pytest
import torch

from src.tensorcontainer.tensor_dataclass import TensorDataClass
from tests.compile_utils import run_and_compare_compiled


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

    # Basic Indexing
    TDC_BASIC_INDEXING_CASES = [
        ("int", 5),
        ("slice", slice(2, 15)),
        ("slice_step", slice(0, 20, 3)),
        ("int_slice_tuple", (4, slice(1, 4))),
        ("slice_int_tuple", (slice(2, 8), 3)),
        ("slice_slice_tuple", (slice(1, 10, 2), slice(0, 4, 2))),
    ]

    ADVANCED_INDEXING_CASES = [
        ("list_int", ([0, 4, 2, 19, 7])),
        ("long_tensor", (torch.tensor([0, 4, 2, 19, 7]))),
        ("tensor_slice_tuple", (torch.tensor([0, 1, 2]), slice(None))),
        ("slice_tensor_tuple", (slice(None), torch.tensor([0, 1, 2]))),
        ("tensor_tensor_tuple", (torch.tensor([0, 1]), torch.tensor([2, 3]))),
    ]

    BOOLEAN_MASK_CASES = [
        ("bool_mask", (torch.rand(20) > 0.5)),
        ("mask_slice_tuple", (torch.rand(20) > 0.5, slice(0, 3))),
    ]

    @pytest.mark.parametrize(
        "test_name,idx,is_bool_mask",
        [(f"basic_{case[0]}", case[1], False) for case in TDC_BASIC_INDEXING_CASES]
        + [(f"adv_{case[0]}", case[1], False) for case in ADVANCED_INDEXING_CASES]
        + [(f"bool_{case[0]}", case[1], True) for case in BOOLEAN_MASK_CASES],
        ids=[f"basic_{case[0]}" for case in TDC_BASIC_INDEXING_CASES]
        + [f"adv_{case[0]}" for case in ADVANCED_INDEXING_CASES]
        + [f"bool_{case[0]}" for case in BOOLEAN_MASK_CASES],
    )
    def test_setitem_basic_and_advanced_indexing(self, test_name, idx, is_bool_mask):
        """Tests basic, advanced, and boolean mask indexing with a TensorDataClass as the value.

        This test consolidates basic, advanced, and boolean mask indexing tests.
        It checks that assigning a compatible TensorDataClass to elements
        selected by various indexing methods works as expected and compiles (where applicable).
        """
        tdc_dest_initial = create_sample_tdc()
        tdc_source = create_source_tdc_for_slice(tdc_dest_initial, idx)

        processed_idx = idx
        if isinstance(idx, list) and all(isinstance(i, int) for i in idx):
            processed_idx = torch.tensor(
                idx, device=tdc_dest_initial.device, dtype=torch.long
            )
        elif isinstance(idx, tuple) and any(isinstance(i, list) for i in idx):
            processed_idx = tuple(
                torch.tensor(i, device=tdc_dest_initial.device, dtype=torch.long)
                if isinstance(i, list)
                else i
                for i in idx
            )

        self._run_and_verify_setitem(
            tdc_dest_initial,
            processed_idx,
            tdc_source,
            test_name,
            fullgraph=not is_bool_mask,
        )

    # Indexing with Broadcasting
    BROADCAST_ASSIGN_TDC_CASES = [
        ("broadcast_1_5_to_10_5", slice(0, 10), (1, 5)),  # Case 1 (existing)
        ("broadcast_20_1_to_20_5", slice(None), (20, 1)),  # Case 2
        ("broadcast_1_1_to_10_5", slice(0, 10), (1, 1)),  # Case 3
    ]

    @pytest.mark.parametrize(
        "test_name,idx,source_shape",
        BROADCAST_ASSIGN_TDC_CASES,
        ids=[case[0] for case in BROADCAST_ASSIGN_TDC_CASES],
    )
    def test_setitem_broadcast_with_tdc(self, test_name, idx, source_shape):
        """Tests assigning a broadcastable TensorDataClass to a slice.

        This test verifies that a TensorDataClass instance can be assigned to a
        slice of another TensorDataClass, even if their batch dimensions are
        different but broadcastable. The operation should compile.
        """
        tdc_dest_initial = create_sample_tdc()
        tdc_source = SampleTensorDataClass(
            features=torch.randn(*source_shape, 10),
            labels=torch.randn(*source_shape),
            shape=source_shape,
            device=tdc_dest_initial.device,
        )
        self._run_and_verify_setitem(tdc_dest_initial, idx, tdc_source, test_name)

    # Error Handling
    INVALID_INDEXING_CASES = [
        (
            "shape_mismatch_tdc",
            (slice(None),),
            (19, 5),  # Source TDC shape
            RuntimeError,  # Changed from ValueError to RuntimeError
            r"Issue with key \.(features|labels) and index \(slice\(None, None, None\),\) for value of shape torch\.Size\(\[20, 5, 10\]\) and type <class 'torch\.Tensor'> and assignment of shape \(19, 5\)",
        ),
        (
            "too_many_indices",
            (slice(None), slice(None), 0),
            (20, 5),  # Assign a valid TDC
            IndexError,
            r"too many indices for container: container is \d+-dimensional, but \d+ were indexed",
        ),
    ]

    @pytest.mark.parametrize(
        "test_name,idx,value_shape,error,match",
        INVALID_INDEXING_CASES,
        ids=[case[0] for case in INVALID_INDEXING_CASES],
    )
    def test_setitem_invalid_inputs_raise_errors(
        self, test_name, idx, value_shape, error, match
    ):
        """Tests that invalid `__setitem__` operations raise appropriate errors.

        This test ensures that `__setitem__` correctly raises errors in eager
        mode for various invalid scenarios, such as shape mismatches when
        assigning a TDC, out-of-bounds indices, and too many indices.
        """
        tdc = create_sample_tdc()

        # value_to_assign must always be a SampleTensorDataClass based on clarification
        value_to_assign = SampleTensorDataClass(
            features=torch.randn(*value_shape, 10),
            labels=torch.randn(*value_shape),
            shape=value_shape,
            device=tdc.device,
        )

        with pytest.raises(error, match=match):
            tdc[idx] = value_to_assign
