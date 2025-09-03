import pytest
import re
import torch
from tensorcontainer.utils import assert_pytrees_equal
from tensorcontainer.tensor_dict import TensorDict
from tensorcontainer.tensor_dataclass import TensorDataClass

# Simplified test constants - using consistent tensor shapes and values
TENSOR_A = torch.tensor([1, 2])  # Standard tensor for most tests
TENSOR_B = torch.tensor([3, 4])  # Alternative tensor with same shape
TENSOR_BATCH = torch.randn(2, 3)  # For batch dimension tests

# Expected error messages for exact matching
BATCH_SHAPES_ERROR = """operation expects each item to have the same structure, but item 0 and item 1 differ
  at path items[0]: TensorDictPytreeContext(keys=('x',), event_ndims=(1,), shape_context=torch.Size([2]), device_context=None, metadata={})
  at path items[1]: TensorDictPytreeContext(keys=('x',), event_ndims=(0,), shape_context=torch.Size([2, 3]), device_context=None, metadata={})"""

DEVICE_DIFF_ERROR = """operation expects each item to have the same structure, but item 0 and item 1 differ
  at path items[0]: TensorDictPytreeContext(keys=('x',), event_ndims=(1,), shape_context=torch.Size([]), device_context=device(type='cpu'), metadata={})
  at path items[1]: TensorDictPytreeContext(keys=('x',), event_ndims=(1,), shape_context=torch.Size([]), device_context=None, metadata={})"""

LIST_VS_TUPLE_ERROR = "operation expects each item to have equal type, but got list at entry 0 and tuple at entry 1"

NESTING_DIFF_ERROR = "Key paths differ: ((SequenceKey(idx=1),), (SequenceKey(idx=0),))"

DIFFERENT_KEYS_ERROR = """operation expects each item to have the same structure, but item 0 and item 1 differ
  at path items[0]: TensorDictPytreeContext(keys=('a', 'b'), event_ndims=(1, 1), shape_context=torch.Size([]), device_context=None, metadata={})
  at path items[1]: TensorDictPytreeContext(keys=('a', 'b', 'c'), event_ndims=(1, 1, 1), shape_context=torch.Size([]), device_context=None, metadata={})"""

EMPTY_VS_POPULATED_ERROR = """operation expects each item to have the same structure, but item 0 and item 1 differ
  at path items[0]: TensorDictPytreeContext(keys=(), event_ndims=(), shape_context=torch.Size([]), device_context=None, metadata={})
  at path items[1]: TensorDictPytreeContext(keys=('x',), event_ndims=(1,), shape_context=torch.Size([]), device_context=None, metadata={})"""

NESTED_VS_FLAT_ERROR = """operation expects each item to have the same structure, but item 0 and item 1 differ
  at path items[0]: TensorDictPytreeContext(keys=('a', 'b'), event_ndims=(1, 1), shape_context=torch.Size([]), device_context=None, metadata={})
  at path items[1]: TensorDictPytreeContext(keys=('a', 'b'), event_ndims=(0, 1), shape_context=torch.Size([]), device_context=None, metadata={})"""

NESTED_MISMATCHED_ERROR = """operation expects each item to have the same structure, but item 0 and item 1 differ
  at path items[0]['outer']: TensorDictPytreeContext(keys=('key1', 'key2'), event_ndims=(1, 1), shape_context=torch.Size([]), device_context=None, metadata={})
  at path items[1]['outer']: TensorDictPytreeContext(keys=('key1', 'key3'), event_ndims=(1, 1), shape_context=torch.Size([]), device_context=None, metadata={})"""

DATACLASS_FIELD_ERROR = "operation expects each item to have equal type, but got SimpleDataClass at entry 0 and DifferentFieldDataClass at entry 1"

DATACLASS_BATCH_ERROR = """operation expects each item to have the same structure, but item 0 and item 1 differ
  at path items[0]: (['a', 'b'], (1, 1), {}, device(type='cpu'))
  at path items[1]: (['a', 'b'], (0, 0), {}, device(type='cpu'))"""

TENSORDICT_VS_LIST_ERROR = "operation expects each item to have equal type, but got TensorDict at entry 0 and list at entry 1"

LIST_VS_DICT_ERROR = "operation expects each item to have equal type, but got list at entry 0 and dict at entry 1"


# Test dataclass definitions
class SimpleDataClass(TensorDataClass):
    a: torch.Tensor
    b: torch.Tensor


class DifferentFieldDataClass(TensorDataClass):
    x: torch.Tensor
    y: torch.Tensor


class NestedDataClass(TensorDataClass):
    outer: SimpleDataClass
    flat: torch.Tensor


class TestBasicFunctionality:
    """Test basic functionality of assert_pytrees_equal function."""

    def test_empty_list(self):
        """Empty list should pass without error."""
        assert_pytrees_equal([])

    def test_single_tree(self):
        """Single tree should pass without error."""
        td = TensorDict({"a": TENSOR_A, "b": TENSOR_B}, shape=())
        assert_pytrees_equal([td])

    def test_same_tensor_objects(self):
        """Trees with same tensor objects should have equal contexts."""
        td1 = TensorDict({"x": TENSOR_A, "y": TENSOR_B}, shape=())
        td2 = TensorDict({"x": TENSOR_A, "y": TENSOR_B}, shape=())
        assert_pytrees_equal([td1, td2])

    def test_different_tensor_values_same_structure(self):
        """Trees with different tensor values but same structure should have equal contexts."""
        tensor_alt1 = torch.tensor([5, 6])  # Different values, same shape as TENSOR_A
        tensor_alt2 = torch.tensor([7, 8])  # Different values, same shape as TENSOR_B

        td1 = TensorDict({"x": TENSOR_A, "y": TENSOR_B}, shape=())
        td2 = TensorDict({"x": tensor_alt1, "y": tensor_alt2}, shape=())
        assert_pytrees_equal([td1, td2])

    def test_different_batch_shapes_should_fail(self):
        """Trees with different tensor container shapes should fail."""
        td1 = TensorDict({"x": TENSOR_BATCH}, shape=(2,))
        td2 = TensorDict({"x": TENSOR_BATCH}, shape=(2, 3))

        with pytest.raises(RuntimeError, match=f"^{re.escape(BATCH_SHAPES_ERROR)}$"):
            assert_pytrees_equal([td1, td2])

    def test_different_devices_should_fail(self):
        """Trees with different device contexts should fail."""
        td1 = TensorDict({"x": TENSOR_A}, shape=(), device="cpu")
        td2 = TensorDict({"x": TENSOR_A}, shape=(), device=None)

        with pytest.raises(RuntimeError, match=f"^{re.escape(DEVICE_DIFF_ERROR)}$"):
            assert_pytrees_equal([td1, td2])


class TestTensorDictStructures:
    """Test cases for TensorDict PyTree structures."""

    def test_identical_structures(self):
        """Identical TensorDict structures should pass."""
        td1 = TensorDict({"a": TENSOR_A, "b": TENSOR_B}, shape=())
        td2 = TensorDict({"a": TENSOR_A, "b": TENSOR_B}, shape=())
        assert_pytrees_equal([td1, td2])

    def test_nested_structures(self):
        """Nested TensorDict structures with same layout should pass."""
        nested_inner = TensorDict({"inner1": TENSOR_A, "inner2": TENSOR_B}, shape=())

        td1 = TensorDict({"outer": nested_inner, "flat": TENSOR_A}, shape=())
        td2 = TensorDict({"outer": nested_inner, "flat": TENSOR_A}, shape=())
        assert_pytrees_equal([td1, td2])

    def test_different_keys_should_fail(self):
        """TensorDicts with different keys should fail."""
        td1 = TensorDict({"a": TENSOR_A, "b": TENSOR_B}, shape=())
        td2 = TensorDict({"a": TENSOR_A, "b": TENSOR_B, "c": TENSOR_A}, shape=())

        with pytest.raises(RuntimeError, match=f"^{re.escape(DIFFERENT_KEYS_ERROR)}$"):
            assert_pytrees_equal([td1, td2])

    def test_empty_vs_populated_should_fail(self):
        """Empty vs populated containers should fail."""
        td_empty = TensorDict({}, shape=())
        td_populated = TensorDict({"x": TENSOR_A}, shape=())

        with pytest.raises(RuntimeError, match=f"^{re.escape(EMPTY_VS_POPULATED_ERROR)}$"):
            assert_pytrees_equal([td_empty, td_populated])

    def test_nested_vs_flat_should_fail(self):
        """Nested vs flat structures with same keys should fail."""
        td_flat = TensorDict({"a": TENSOR_A, "b": TENSOR_B}, shape=())
        td_nested = TensorDict(
            {
                "a": TensorDict({"nested": TENSOR_A}, shape=()),
                "b": TENSOR_B,
            },
            shape=(),
        )

        with pytest.raises(RuntimeError, match=f"^{re.escape(NESTED_VS_FLAT_ERROR)}$"):
            assert_pytrees_equal([td_flat, td_nested])

    def test_nested_mismatched_inner_keys_should_fail(self):
        """Nested TensorDicts with mismatched inner keys should fail."""
        td1 = TensorDict(
            {
                "outer": TensorDict({"key1": TENSOR_A, "key2": TENSOR_B}, shape=()),
                "flat": TENSOR_A,
            },
            shape=(),
        )
        td2 = TensorDict(
            {
                "outer": TensorDict(
                    {"key1": TENSOR_A, "key3": TENSOR_B}, shape=()
                ),  # key3 instead of key2
                "flat": TENSOR_A,
            },
            shape=(),
        )

        with pytest.raises(RuntimeError, match=f"^{re.escape(NESTED_MISMATCHED_ERROR)}$"):
            assert_pytrees_equal([td1, td2])


class TestListTupleStructures:
    """Test cases for list and tuple PyTree structures."""

    def test_identical_lists(self):
        """Identical list structures should pass."""
        list1 = [TENSOR_A, TENSOR_B]
        list2 = [TENSOR_A, TENSOR_B]
        assert_pytrees_equal([list1, list2])

    def test_identical_tuples(self):
        """Identical tuple structures should pass."""
        tuple1 = (TENSOR_A, TENSOR_B)
        tuple2 = (TENSOR_A, TENSOR_B)
        assert_pytrees_equal([tuple1, tuple2])

    def test_list_vs_tuple_should_fail(self):
        """List vs tuple with same content should fail."""
        list_tree = [TENSOR_A, TENSOR_B]
        tuple_tree = (TENSOR_A, TENSOR_B)

        with pytest.raises(RuntimeError, match=f"^{re.escape(LIST_VS_TUPLE_ERROR)}$"):
            assert_pytrees_equal([list_tree, tuple_tree])

    def test_different_nesting_should_fail(self):
        """Trees with different nesting depths should fail."""
        tree1 = [TENSOR_A, [TENSOR_B]]
        tree2 = [[TENSOR_A], TENSOR_B]

        with pytest.raises(RuntimeError, match=f"^{re.escape(NESTING_DIFF_ERROR)}$"):
            assert_pytrees_equal([tree1, tree2])


class TestTensorDataClassStructures:
    """Test cases for TensorDataClass PyTree structures."""

    def test_identical_dataclass_structures(self):
        """Identical TensorDataClass structures should pass."""
        dc1 = SimpleDataClass(a=TENSOR_A, b=TENSOR_B, shape=(), device="cpu")
        dc2 = SimpleDataClass(a=TENSOR_A, b=TENSOR_B, shape=(), device="cpu")
        assert_pytrees_equal([dc1, dc2])

    def test_nested_dataclass_structures(self):
        """Nested TensorDataClass structures with same layout should pass."""
        outer = SimpleDataClass(a=TENSOR_A, b=TENSOR_B, shape=(), device="cpu")

        dc1 = NestedDataClass(outer=outer, flat=TENSOR_A, shape=(), device="cpu")
        dc2 = NestedDataClass(outer=outer, flat=TENSOR_A, shape=(), device="cpu")
        assert_pytrees_equal([dc1, dc2])

    def test_different_field_names_should_fail(self):
        """TensorDataClasses with different field names should fail."""
        dc1 = SimpleDataClass(a=TENSOR_A, b=TENSOR_B, shape=(), device="cpu")
        dc2 = DifferentFieldDataClass(x=TENSOR_A, y=TENSOR_B, shape=(), device="cpu")

        with pytest.raises(RuntimeError, match=f"^{re.escape(DATACLASS_FIELD_ERROR)}$"):
            assert_pytrees_equal([dc1, dc2])

    def test_different_tensor_values_same_structure(self):
        """DataClass instances with different tensor values but same structure should pass."""
        tensor_alt1 = torch.tensor([10, 20])
        tensor_alt2 = torch.tensor([30, 40])

        dc1 = SimpleDataClass(a=TENSOR_A, b=TENSOR_B, shape=(), device="cpu")
        dc2 = SimpleDataClass(a=tensor_alt1, b=tensor_alt2, shape=(), device="cpu")
        assert_pytrees_equal([dc1, dc2])

    def test_different_batch_shapes_should_fail(self):
        """DataClass instances with different batch shapes should fail."""
        dc1 = SimpleDataClass(a=TENSOR_BATCH, b=TENSOR_BATCH, shape=(2,), device="cpu")
        dc2 = SimpleDataClass(
            a=TENSOR_BATCH, b=TENSOR_BATCH, shape=(2, 3), device="cpu"
        )

        with pytest.raises(RuntimeError, match=f"^{re.escape(DATACLASS_BATCH_ERROR)}$"):
            assert_pytrees_equal([dc1, dc2])


class TestMixedStructures:
    """Test cases for mixed PyTree node types."""

    def test_tensordict_vs_list_should_fail(self):
        """TensorDict vs list should fail due to type mismatch."""
        td = TensorDict({"a": TENSOR_A}, shape=())
        list_tree = [TENSOR_A, TENSOR_B]

        with pytest.raises(RuntimeError, match=f"^{re.escape(TENSORDICT_VS_LIST_ERROR)}$"):
            assert_pytrees_equal([td, list_tree])

    def test_list_vs_dict_should_fail(self):
        """List vs dict should fail due to type mismatch."""
        list_tree = [TENSOR_A, TENSOR_B]
        dict_tree = {"0": TENSOR_A, "1": TENSOR_B}

        with pytest.raises(RuntimeError, match=f"^{re.escape(LIST_VS_DICT_ERROR)}$"):
            assert_pytrees_equal([list_tree, dict_tree])

    def test_tensordict_vs_dataclass_should_fail(self):
        """TensorDict vs TensorDataClass should fail despite similar structure."""
        td = TensorDict({"a": TENSOR_A, "b": TENSOR_B}, shape=())
        dc = SimpleDataClass(a=TENSOR_A, b=TENSOR_B, shape=(), device="cpu")

        with pytest.raises(RuntimeError):
            assert_pytrees_equal([td, dc])
