from typing import Any, List, Optional

import pytest
import torch
from torch._dynamo import exc as dynamo_exc

from rtd.tensor_dataclass import TensorDataClass
from tests.conftest import skipif_no_compile
from tests.tensor_dataclass.conftest import (
    NestedTensorDataClass,
    compute_stack_shape,
)
from tests.tensor_dict.compile_utils import run_and_compare_compiled


class TestStack:
    """Test suite for torch.stack operations on TensorDataclass instances."""

    @staticmethod
    def _stack_operation(tensor_dataclass_list, dim_arg):
        """Helper method for stack operations."""
        return torch.stack(tensor_dataclass_list, dim=dim_arg)

    def _create_test_pair(self, nested_tensor_data_class):
        """Creates a pair of dataclass instances for testing."""
        td1 = nested_tensor_data_class
        td2 = td1.clone()

        # Normalize to prevent pytree context mismatches
        td2.device = td1.device
        td2.shape = td1.shape
        td2.tensor_data_class.device = td1.tensor_data_class.device
        td2.tensor_data_class.shape = td1.tensor_data_class.shape

        return td1, td2

    def _verify_stack_result(self, stacked_td, td1, td2, dim):
        """Helper method to verify stack operation results."""
        assert isinstance(stacked_td, NestedTensorDataClass)

        original_batch_shape = td1.shape
        expected_batch_shape = compute_stack_shape(original_batch_shape, dim)
        event_shape = td1.tensor.shape[len(original_batch_shape) :]
        expected_tensor_shape = expected_batch_shape + event_shape

        assert stacked_td.shape == expected_batch_shape
        assert stacked_td.tensor.shape == expected_tensor_shape
        assert stacked_td.tensor_data_class.tensor.shape == expected_tensor_shape

        assert stacked_td.meta_data == td1.meta_data

        normalized_dim = dim if dim >= 0 else dim + len(original_batch_shape) + 1
        slicer: List[Any] = [slice(None)] * len(expected_tensor_shape)

        slicer[normalized_dim] = 0
        assert torch.equal(stacked_td.tensor[tuple(slicer)], td1.tensor)

        slicer[normalized_dim] = 1
        assert torch.equal(stacked_td.tensor[tuple(slicer)], td2.tensor)

    @pytest.mark.parametrize("dim", [0, 1, 2, -1, -2, -3])  # Valid batch dimensions
    def test_stack_valid_dims(self, nested_tensor_data_class, dim):
        """Test stack operation with valid dimensions."""
        td1, td2 = self._create_test_pair(nested_tensor_data_class)
        stacked_td = self._stack_operation([td1, td2], dim)
        self._verify_stack_result(stacked_td, td1, td2, dim)

    @pytest.mark.parametrize("dim", [3, -4])  # Invalid dimensions
    def test_stack_invalid_dim_raises(self, nested_tensor_data_class, dim):
        """Test stack operation raises with invalid dimensions."""
        td1, td2 = self._create_test_pair(nested_tensor_data_class)
        with pytest.raises(IndexError, match="Dimension out of range"):
            self._stack_operation([td1, td2], dim)

    def test_stack_inconsistent_shapes_raises(self, nested_tensor_data_class):
        """Test that stacking with inconsistent shapes raises an error."""
        td1, td2 = self._create_test_pair(nested_tensor_data_class)

        # Create an inconsistent shape for one of the tensors.
        td2.shape = (td1.shape[0] + 1, td1.shape[1])

        with pytest.raises(
            ValueError, match="stack expects each TensorContainer to be equal size"
        ):
            self._stack_operation([td1, td2], 0)

    def test_stack_inconsistent_meta_data_raises(self, nested_tensor_data_class):
        """Test that stacking with inconsistent meta data raises an error."""
        td1, td2 = self._create_test_pair(nested_tensor_data_class)
        td2.meta_data = "different_meta"
        with pytest.raises(ValueError, match="Node context mismatch"):
            self._stack_operation([td1, td2], 0)

    @skipif_no_compile
    def test_stack_compile(self, nested_tensor_data_class):
        """Tests that a function using torch.stack with TensorDataclass can be torch.compiled."""
        td1, td2 = self._create_test_pair(nested_tensor_data_class)
        td2.tensor.mul_(2)
        td2.tensor_data_class.tensor.mul_(2)
        run_and_compare_compiled(self._stack_operation, [td1, td2], 0)

    @skipif_no_compile
    def test_stack_compile_invalid_dim_raises(self, nested_tensor_data_class):
        """Test stack operation raises with invalid dimensions in compile mode."""
        td1, td2 = self._create_test_pair(nested_tensor_data_class)
        compiled_stack_op = torch.compile(self._stack_operation, fullgraph=True)
        with pytest.raises(dynamo_exc.Unsupported) as excinfo:
            compiled_stack_op([td1, td2], 3)  # Invalid dimension
        assert "IndexError" in str(excinfo.value)

    def test_stack_empty_list_raises(self):
        """Test that torch.stack on an empty list raises a RuntimeError."""
        with pytest.raises(RuntimeError, match="stack expects a non-empty TensorList"):
            self._stack_operation([], 0)

    def test_stack_mixed_optional_raises(self):
        """Test that stacking with mixed None and Tensor for an optional field raises."""

        class OptionalStack(TensorDataClass):
            shape: tuple
            device: Optional[torch.device]
            a: torch.Tensor
            b: Optional[torch.Tensor] = None

        td1 = OptionalStack(
            shape=(3,), device=torch.device("cpu"), a=torch.randn(3), b=torch.ones(3)
        )
        td2 = OptionalStack(
            shape=(3,), device=torch.device("cpu"), a=torch.randn(3), b=None
        )

        with pytest.raises(ValueError, match="Node arity mismatch"):
            torch.stack([td1, td2], dim=0)  # type: ignore
