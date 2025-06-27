import pytest
import torch
from typing import Tuple

from rtd.tensor_dataclass import TensorDataClass
from rtd.tensor_dict import TensorDict

# Define compile_kwargs locally as it's not exported from compile_utils
compile_kwargs = {"fullgraph": True}


class TensorDataClassWithTensorDict(TensorDataClass):
    """
    A TensorDataClass that contains a TensorDict as a child attribute.
    Used for testing the interaction between TensorDataClass and TensorDict.
    """

    nested_td: TensorDict
    other_tensor: torch.Tensor


def _make_tdc_with_td(
    batch_size: Tuple[int, ...], device: torch.device, requires_grad: bool = False
) -> TensorDataClassWithTensorDict:
    """Helper function to create an instance of TensorDataClassWithTensorDict."""
    # Create a simple, non-nested TensorDict for testing
    nested_td = TensorDict(
        {
            "a": torch.randn(
                *batch_size, 3, device=device, requires_grad=requires_grad
            ),
            "b": torch.randn(
                *batch_size, 4, device=device, requires_grad=requires_grad
            ),
        },
        shape=batch_size,
    )
    other_tensor = torch.randn(
        *batch_size, 5, device=device, requires_grad=requires_grad
    )
    return TensorDataClassWithTensorDict(
        nested_td=nested_td, other_tensor=other_tensor, shape=batch_size, device=device
    )


def _assert_td_close(td1, td2):
    """Recursively compares two TensorDict objects."""
    assert isinstance(td1, TensorDict)
    assert isinstance(td2, TensorDict)
    assert td1.shape == td2.shape
    if td1.device and td2.device:
        assert td1.device.type == td2.device.type
    assert set(td1.keys()) == set(td2.keys())

    for key in td1.keys():
        v1 = td1[key]
        v2 = td2[key]
        if isinstance(v1, TensorDict):
            _assert_td_close(v1, v2)
        else:
            torch.testing.assert_close(v1, v2)


def _assert_tdc_with_td_close(tdc1, tdc2):
    """Compares two TensorDataClassWithTensorDict objects."""
    assert isinstance(tdc1, TensorDataClassWithTensorDict)
    assert isinstance(tdc2, TensorDataClassWithTensorDict)
    assert tdc1.shape == tdc2.shape
    if tdc1.device and tdc2.device:
        assert tdc1.device.type == tdc2.device.type

    _assert_td_close(tdc1.nested_td, tdc2.nested_td)
    torch.testing.assert_close(tdc1.other_tensor, tdc2.other_tensor)


class TestGetItem:
    @pytest.mark.parametrize(
        "batch_size, idx",
        [
            ((1,), 0),
            ((1,), slice(0, 1)),
            ((2, 3), 0),
            ((2, 3), torch.tensor([0, 1])),
            ((2, 3), slice(0, 1)),
        ],
    )
    def test_getitem(self, batch_size, idx, device):
        """
        Tests __getitem__ operation on a TensorDataClass with a nested TensorDict.

        This test ensures that indexing a TensorDataClass containing a TensorDict
        correctly applies the indexing to both the TensorDict child and other
        tensor attributes.

        Example:
            >>> tdc = _make_tdc_with_td((2,), "cpu")
            >>> tdc[0]
        """
        tdc = _make_tdc_with_td(batch_size, device)

        def _getitem_op(tdc_instance, index):
            return tdc_instance[index]

        self._run_and_verify_operation(_getitem_op, tdc, idx)

    def _run_and_verify_operation(self, op, tdc, *args):
        """
        Helper to run an operation in eager and compiled mode and verify results.
        """
        idx = args[0]  # To inspect and decide whether to compile
        result_eager = op(tdc, *args)

        # Temp fix: Skip torch.compile for tensor indexing due to recompile limits
        if not isinstance(idx, torch.Tensor):
            result_compiled = torch.compile(op, fullgraph=compile_kwargs["fullgraph"])(
                tdc, *args
            )

            assert isinstance(result_eager, TensorDataClassWithTensorDict)
            assert isinstance(result_compiled, TensorDataClassWithTensorDict)

            _assert_tdc_with_td_close(result_eager, result_compiled)

        # Verify content
        # Unpack args for indexing due to Python 3.8 syntax limitations
        expected_nested_td = tdc.nested_td[idx]
        expected_other_tensor = tdc.other_tensor[idx]

        _assert_td_close(result_eager.nested_td, expected_nested_td)
        torch.testing.assert_close(result_eager.other_tensor, expected_other_tensor)

        assert result_eager.shape == expected_nested_td.shape
        assert result_eager.device == expected_nested_td.device


class TestDetach:
    @pytest.mark.parametrize("batch_size", [(1,), (2, 3)])
    def test_detach(self, batch_size, device):
        """
        Tests detach operation on a TensorDataClass with a nested TensorDict.

        This test verifies that calling `detach()` on the TensorDataClass
        correctly detaches all tensors, including those within the nested
        TensorDict.
        """
        tdc = _make_tdc_with_td(batch_size, device, requires_grad=True)

        # Verify that requires_grad is initially True for all tensors
        assert tdc.nested_td["a"].requires_grad  # type: ignore
        assert tdc.nested_td["b"].requires_grad  # type: ignore
        assert tdc.other_tensor.requires_grad

        # Define and run the detach operation
        def _detach_op(tdc_instance):
            return tdc_instance.detach()

        self._run_and_verify_operation(_detach_op, tdc)

    def _run_and_verify_operation(self, op, tdc):
        """
        Helper to run an operation in eager and compiled mode and verify results.
        """
        result_eager = op(tdc)
        result_compiled = torch.compile(op, fullgraph=compile_kwargs["fullgraph"])(tdc)

        assert isinstance(result_eager, TensorDataClassWithTensorDict)
        assert isinstance(result_compiled, TensorDataClassWithTensorDict)

        _assert_tdc_with_td_close(result_eager, result_compiled)

        # Verify content and no grad
        assert not result_eager.nested_td["a"].requires_grad  # type: ignore
        assert not result_eager.nested_td["b"].requires_grad  # type: ignore
        assert not result_eager.other_tensor.requires_grad
        assert not result_compiled.nested_td["a"].requires_grad  # type: ignore
        assert not result_compiled.nested_td["b"].requires_grad  # type: ignore
        assert not result_compiled.other_tensor.requires_grad

        _assert_tdc_with_td_close(result_eager, tdc)


class TestView:
    @pytest.mark.parametrize("batch_size", [(2, 4)])
    @pytest.mark.parametrize("new_shape", [(8,), (2, 2, 2)])
    def test_view(self, batch_size, new_shape, device):
        """
        Tests view operation on a TensorDataClass with a nested TensorDict.

        This test checks that the `view()` operation is successfully applied to
        both the nested TensorDict and other tensor attributes of the
        TensorDataClass.
        """
        tdc = _make_tdc_with_td(batch_size, device)

        def _view_op(tdc_instance, shape):
            return tdc_instance.view(*shape)

        self._run_and_verify_operation(_view_op, tdc, new_shape)

    def _run_and_verify_operation(self, op, tdc, *args):
        """
        Helper to run an operation in eager and compiled mode and verify results.
        """
        result_eager = op(tdc, *args)
        result_compiled = torch.compile(op, fullgraph=compile_kwargs["fullgraph"])(
            tdc, *args
        )

        assert isinstance(result_eager, TensorDataClassWithTensorDict)
        assert isinstance(result_compiled, TensorDataClassWithTensorDict)

        _assert_tdc_with_td_close(result_eager, result_compiled)

        # Verify content and shape
        # Unpack args for view due to Python 3.8 syntax limitations
        shape = args[0]
        expected_nested_td = tdc.nested_td.view(*shape)
        event_shape = tdc.other_tensor.shape[len(tdc.shape) :]
        expected_other_tensor = tdc.other_tensor.view(*shape, *event_shape)

        _assert_td_close(result_eager.nested_td, expected_nested_td)
        torch.testing.assert_close(result_eager.other_tensor, expected_other_tensor)

        assert result_eager.shape == expected_nested_td.shape
        assert result_eager.device == expected_nested_td.device


class TestSetItem:
    @pytest.mark.parametrize(
        "leaf_to_modify, expected_error_path",
        [
            ("nested_td.a", "nested_td\\['a'\\]"),
            ("nested_td.b", "nested_td\\['b'\\]"),
            ("other_tensor", "other_tensor"),
        ],
    )
    def test_setitem_invalid_shape_raises_error_with_path(
        self, leaf_to_modify, expected_error_path, device
    ):
        """
        Tests that __setitem__ raises a ValueError with the correct path
        on shape mismatch in a nested TensorDict.
        """
        batch_size = (2,)
        dest_tdc = _make_tdc_with_td(batch_size, device)
        source_tdc = _make_tdc_with_td((), device)  # Batch size of 1 for source

        # Modify the shape of the source tensor to be invalid
        if leaf_to_modify == "nested_td.a":
            source_tdc.nested_td["a"] = torch.randn(99, device=device)
        elif leaf_to_modify == "nested_td.b":
            source_tdc.nested_td["b"] = torch.randn(99, device=device)
        elif leaf_to_modify == "other_tensor":
            source_tdc.other_tensor = torch.randn(99, device=device)

        with pytest.raises(
            ValueError,
            match=f"Assignment failed for leaf at path '{expected_error_path}'",
        ):
            dest_tdc[0] = source_tdc

    @pytest.mark.parametrize(
        "leaf_to_modify, expected_error_path",
        [
            ("nested_td.a", "nested_td\\['a'\\]"),
            ("nested_td.b", "nested_td\\['b'\\]"),
            ("other_tensor", "other_tensor"),
        ],
    )
    def test_setitem_invalid_event_shape_raises_error_with_path(
        self, leaf_to_modify, expected_error_path, device
    ):
        """
        Tests that __setitem__ raises a ValueError with the correct path
        on event shape mismatch in a nested TensorDict.
        """
        batch_size = (2,)
        dest_tdc = _make_tdc_with_td(batch_size, device)
        source_tdc = _make_tdc_with_td((), device)  # Batch size of 1 for source

        # Modify the event shape of the source tensor to be invalid
        if leaf_to_modify == "nested_td.a":
            source_tdc.nested_td["a"] = torch.randn(4, device=device)
        elif leaf_to_modify == "nested_td.b":
            source_tdc.nested_td["b"] = torch.randn(5, device=device)
        elif leaf_to_modify == "other_tensor":
            source_tdc.other_tensor = torch.randn(6, device=device)

        with pytest.raises(
            ValueError,
            match=f"Assignment failed for leaf at path '{expected_error_path}'",
        ):
            dest_tdc[0] = source_tdc

    def test_setitem_replace_nested_td_mismatched_event_shape_raises(self, device):
        """
        Tests that setting a nested TensorDict with a mismatched event shape
        in a child tensor raises a ValueError.
        """
        batch_size = (2,)
        dest_tdc = _make_tdc_with_td(batch_size, device)
        source_tdc = _make_tdc_with_td((), device)

        # Create a new TensorDict where 'a' has a different event shape
        td_wrong_event_shape = TensorDict(
            {
                "a": torch.randn(4, device=device),  # Mismatched event shape
                "b": source_tdc.nested_td["b"].clone(),
            },
            shape=source_tdc.nested_td.shape,
            device=device,
        )
        source_tdc.nested_td = td_wrong_event_shape

        with pytest.raises(
            ValueError,
            match="Assignment failed for leaf at path 'nested_td\\[\\'a\\'\\]'.*",
        ):
            dest_tdc[0] = source_tdc
