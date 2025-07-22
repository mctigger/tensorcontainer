"""
Tests for torch.stack operations on TensorDataClass instances.

This module verifies the behavior of `torch.stack` on `TensorDataClass` objects,
including handling of different dimensions, shapes, metadata, optional fields,
and edge cases.
"""

from typing import Any, List, Optional, cast

import pytest
import torch
from torch import testing
from torch._dynamo import exc as dynamo_exc

from tensorcontainer.tensor_dataclass import TensorDataClass
from tests.compile_utils import run_and_compare_compiled
from tests.conftest import skipif_no_compile
from tests.tensor_dataclass.conftest import (
    NestedTensorDataClass,
    OptionalFieldsTestClass,
    compute_stack_shape,
)


def _stack_operation(tensor_dataclass_list, dim_arg):
    """Helper method for stack operations."""
    return torch.stack(tensor_dataclass_list, dim=dim_arg)


def _create_test_pair(nested_tensor_data_class):
    """Creates a pair of dataclass instances for testing."""
    td1 = nested_tensor_data_class
    td2 = td1.clone()

    # Normalize to prevent pytree context mismatches
    td2.device = td1.device
    td2.shape = td1.shape
    td2.tensor_data_class.device = td1.tensor_data_class.device
    td2.tensor_data_class.shape = td1.tensor_data_class.shape

    return td1, td2


class TestStackGeneral:
    """
    Tests the general `torch.stack` operation on `TensorDataClass` instances.

    This suite verifies that:
    - Stacking works correctly with valid dimensions.
    - Errors are raised for invalid dimensions.
    - Errors are raised for inconsistent tensor shapes.
    - Errors are raised for inconsistent metadata.
    - Stacking an empty list raises a `RuntimeError`.
    """

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
        testing.assert_close(stacked_td.tensor[tuple(slicer)], td1.tensor)

        slicer[normalized_dim] = 1
        testing.assert_close(stacked_td.tensor[tuple(slicer)], td2.tensor)

    @pytest.mark.parametrize("dim", [0, 1, 2, -1, -2, -3])  # Valid batch dimensions
    def test_stack_valid_dims(self, nested_tensor_data_class, dim):
        """Tests that stacking works correctly across valid dimensions."""
        td1, td2 = _create_test_pair(nested_tensor_data_class)
        stacked_td = _stack_operation([td1, td2], dim)
        self._verify_stack_result(stacked_td, td1, td2, dim)

    @pytest.mark.parametrize("dim", [3, -4])  # Invalid dimensions
    def test_stack_invalid_dim_raises(self, nested_tensor_data_class, dim):
        """Tests that stacking with an invalid dimension raises an IndexError."""
        td1, td2 = _create_test_pair(nested_tensor_data_class)
        with pytest.raises(IndexError, match="Dimension out of range"):
            _stack_operation([td1, td2], dim)

    def test_stack_inconsistent_shapes_raises(self, nested_tensor_data_class):
        """Tests that stacking instances with inconsistent shapes raises a ValueError."""
        td1, td2 = _create_test_pair(nested_tensor_data_class)
        # Create an inconsistent shape for one of the tensors.
        td2.shape = (td1.shape[0] + 1, td1.shape[1])
        with pytest.raises(
            ValueError, match="stack expects each TensorContainer to be equal size"
        ):
            _stack_operation([td1, td2], 0)

    def test_stack_inconsistent_meta_data_raises(self, nested_tensor_data_class):
        """Tests that stacking instances with inconsistent metadata raises a ValueError."""
        td1, td2 = _create_test_pair(nested_tensor_data_class)
        td2.meta_data = "different_meta"
        with pytest.raises(ValueError, match="Node context mismatch"):
            _stack_operation([td1, td2], 0)

    def test_stack_empty_list_raises(self):
        """Tests that `torch.stack` on an empty list raises a RuntimeError."""
        with pytest.raises(RuntimeError, match="stack expects a non-empty TensorList"):
            _stack_operation([], 0)


class TestStackOptionalFields:
    """
    Tests stacking behavior with `Optional` and `default_factory` fields.

    This suite verifies that:
    - Stacking is successful when an optional field is `None` in all instances.
    - Stacking is successful when an optional field is a `Tensor` in all instances.
    - Stacking is successful for tensor fields with a `default_factory`.
    - An error is raised when stacking mixed `None` and `Tensor` optional fields.
    """

    def test_stack_with_optional_tensor_as_none(self):
        """Tests stacking when an optional tensor field is consistently None."""

        def _test_stack_none():
            data1 = OptionalFieldsTestClass(
                shape=(2, 3, 4),
                device=None,
                obs=torch.ones(2, 3, 4, 32, 32),
                reward=None,
                info=["step1"],
            )
            data2 = data1.clone()
            return cast(OptionalFieldsTestClass, torch.stack([data1, data2], dim=0))  # type: ignore

        stacked_data, _ = run_and_compare_compiled(_test_stack_none)
        assert stacked_data.obs.shape == (2, 2, 3, 4, 32, 32)
        assert stacked_data.reward is None
        assert stacked_data.info == ["step1"]
        assert stacked_data.optional_meta is None
        assert stacked_data.optional_meta_val == "value"
        assert stacked_data.shape == (2, 2, 3, 4)

    def test_stack_with_optional_tensor_as_tensor(self):
        """Tests stacking when an optional field is a tensor in all instances."""

        def _test_stack_tensor():
            data1 = OptionalFieldsTestClass(
                shape=(2, 3, 4),
                device=None,
                obs=torch.ones(2, 3, 4, 32, 32),
                reward=torch.ones(2, 3, 4),
            )
            data2 = data1.clone()
            assert data2.reward is not None
            data2.reward.mul_(2)
            return cast(OptionalFieldsTestClass, torch.stack([data1, data2], dim=0))  # type: ignore

        stacked_data, _ = run_and_compare_compiled(_test_stack_tensor)
        assert stacked_data.obs.shape == (2, 2, 3, 4, 32, 32)
        assert stacked_data.reward is not None
        assert stacked_data.reward.shape == (2, 2, 3, 4)
        testing.assert_close(stacked_data.reward[0], torch.ones(2, 3, 4))
        testing.assert_close(stacked_data.reward[1], torch.ones(2, 3, 4) * 2)
        assert stacked_data.info == []
        assert stacked_data.optional_meta is None
        assert stacked_data.optional_meta_val == "value"
        assert stacked_data.shape == (2, 2, 3, 4)

    def test_stack_mixed_optional_raises(self):
        """Tests that stacking mixed None and Tensor for an optional field raises."""

        class OptionalStack(TensorDataClass):
            a: torch.Tensor
            b: Optional[torch.Tensor] = None

        td1 = OptionalStack(
            shape=(2, 3), device=torch.device("cpu"), a=torch.randn(2, 3), b=torch.ones(2, 3)
        )
        td2 = OptionalStack(
            shape=(2, 3), device=torch.device("cpu"), a=torch.randn(2, 3), b=None
        )

        with pytest.raises(ValueError, match="Node arity mismatch"):
            torch.stack([td1, td2], dim=0)  # type: ignore

    @skipif_no_compile
    def test_default_factory_tensor_stack(self):
        """Tests that tensor fields with `default_factory` are stacked correctly."""

        def _test_default_factory_tensor():
            data1 = OptionalFieldsTestClass(
                shape=(2, 3, 4),
                device=None,
                obs=torch.ones(2, 3, 4, 32, 32),
                reward=None,
                info=["step1"],
            )
            data2 = data1.clone()
            stacked_data = cast(
                OptionalFieldsTestClass,
                torch.stack([data1, data2], dim=0),  # type: ignore
            )
            return data1, stacked_data

        (data1, stacked_data), _ = run_and_compare_compiled(
            _test_default_factory_tensor
        )
        assert data1.default_tensor.shape == (2, 3, 4)
        testing.assert_close(data1.default_tensor, torch.zeros(2, 3, 4))
        assert stacked_data.default_tensor.shape == (2, 2, 3, 4)
        testing.assert_close(stacked_data.default_tensor, torch.zeros(2, 2, 3, 4))


@skipif_no_compile
class TestStackCompile:
    """
    Tests `torch.compile` behavior for `torch.stack` on `TensorDataClass`.

    This suite verifies that:
    - Stacking operations can be successfully compiled with `torch.compile`.
    - Compiled stacking raises an error for invalid dimensions.
    """

    def test_stack_compile(self, nested_tensor_data_class):
        """Tests that `torch.stack` can be compiled successfully."""
        td1, td2 = _create_test_pair(nested_tensor_data_class)
        td2.tensor.mul_(2)
        td2.tensor_data_class.tensor.mul_(2)
        run_and_compare_compiled(_stack_operation, [td1, td2], 0)

    def test_stack_compile_invalid_dim_raises(self, nested_tensor_data_class):
        """Tests that compiled stacking raises an error for invalid dimensions."""
        td1, td2 = _create_test_pair(nested_tensor_data_class)
        compiled_stack_op = torch.compile(_stack_operation, fullgraph=True)
        with pytest.raises(dynamo_exc.Unsupported) as excinfo:
            compiled_stack_op([td1, td2], 3)  # Invalid dimension
        assert "IndexError" in str(excinfo.value)
