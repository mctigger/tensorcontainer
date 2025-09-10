import torch
from tensorcontainer.utils import (
    diagnose_pytree_structure_mismatch,
    ContextMismatch,
    TypeMismatch,
    KeyPathMismatch,
)
from tensorcontainer.tensor_dict import TensorDict
from tensorcontainer.tensor_dataclass import TensorDataClass

# Simplified test constants - using consistent tensor shapes and values
TENSOR_A = torch.tensor([1, 2])  # Standard tensor for most tests
TENSOR_B = torch.tensor([3, 4])  # Alternative tensor with same shape
TENSOR_BATCH = torch.randn(2, 3)  # For batch dimension tests

# Simplified test constants - using consistent tensor shapes and values


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
    """Test basic functionality of diagnose_pytree_structure_mismatch function."""

    def test_empty_list(self):
        """Empty list should pass without error."""
        # Empty case - just pass empty list as single argument
        result = diagnose_pytree_structure_mismatch([])
        expected = None
        assert result == expected

    def test_single_tree(self):
        """Single tree should pass without error."""
        td = TensorDict({"a": TENSOR_A, "b": TENSOR_B}, shape=())
        result = diagnose_pytree_structure_mismatch(td)
        expected = None
        assert result == expected

    def test_same_tensor_objects(self):
        """Trees with same tensor objects should have equal contexts."""
        td1 = TensorDict({"x": TENSOR_A, "y": TENSOR_B}, shape=())
        td2 = TensorDict({"x": TENSOR_A, "y": TENSOR_B}, shape=())
        result = diagnose_pytree_structure_mismatch(td1, td2)
        expected = None
        assert result == expected

    def test_different_tensor_values_same_structure(self):
        """Trees with different tensor values but same structure should have equal contexts."""
        tensor_alt1 = torch.tensor([5, 6])  # Different values, same shape as TENSOR_A
        tensor_alt2 = torch.tensor([7, 8])  # Different values, same shape as TENSOR_B

        td1 = TensorDict({"x": TENSOR_A, "y": TENSOR_B}, shape=())
        td2 = TensorDict({"x": tensor_alt1, "y": tensor_alt2}, shape=())
        result = diagnose_pytree_structure_mismatch(td1, td2)
        expected = None
        assert result == expected

    def test_different_batch_shapes_should_fail(self):
        """Trees with different tensor container shapes should fail."""
        td1 = TensorDict({"x": TENSOR_BATCH}, shape=(2,))
        td2 = TensorDict({"x": TENSOR_BATCH}, shape=(2, 3))

        result = diagnose_pytree_structure_mismatch(td1, td2)
        expected = ContextMismatch(
            expected_context=td1._pytree_flatten()[1],
            actual_context=td2._pytree_flatten()[1],
            entry_index=1,
            key_path=(),
        )
        assert result == expected

    def test_different_devices_should_fail(self):
        """Trees with different device contexts should fail."""
        td1 = TensorDict({"x": TENSOR_A}, shape=(), device="cpu")
        td2 = TensorDict({"x": TENSOR_A}, shape=(), device=None)

        result = diagnose_pytree_structure_mismatch(td1, td2)
        expected = ContextMismatch(
            expected_context=td1._pytree_flatten()[1],
            actual_context=td2._pytree_flatten()[1],
            entry_index=1,
            key_path=(),
        )
        assert result == expected


class TestTensorDictStructures:
    """Test cases for TensorDict PyTree structures."""

    def test_identical_structures(self):
        """Identical TensorDict structures should pass."""
        td1 = TensorDict({"a": TENSOR_A, "b": TENSOR_B}, shape=())
        td2 = TensorDict({"a": TENSOR_A, "b": TENSOR_B}, shape=())
        result = diagnose_pytree_structure_mismatch(td1, td2)
        expected = None
        assert result == expected

    def test_nested_structures(self):
        """Nested TensorDict structures with same layout should pass."""
        nested_inner = TensorDict({"inner1": TENSOR_A, "inner2": TENSOR_B}, shape=())

        td1 = TensorDict({"outer": nested_inner, "flat": TENSOR_A}, shape=())
        td2 = TensorDict({"outer": nested_inner, "flat": TENSOR_A}, shape=())
        result = diagnose_pytree_structure_mismatch(td1, td2)
        expected = None
        assert result == expected

    def test_different_keys_should_fail(self):
        """TensorDicts with different keys should fail."""
        td1 = TensorDict({"a": TENSOR_A, "b": TENSOR_B}, shape=())
        td2 = TensorDict({"a": TENSOR_A, "b": TENSOR_B, "c": TENSOR_A}, shape=())

        result = diagnose_pytree_structure_mismatch(td1, td2)
        expected = ContextMismatch(
            expected_context=td1._pytree_flatten()[1],
            actual_context=td2._pytree_flatten()[1],
            entry_index=1,
            key_path=(),
        )
        assert result == expected

    def test_empty_vs_populated_should_fail(self):
        """Empty vs populated containers should fail."""
        td_empty = TensorDict({}, shape=())
        td_populated = TensorDict({"x": TENSOR_A}, shape=())

        result = diagnose_pytree_structure_mismatch(td_empty, td_populated)
        expected = ContextMismatch(
            expected_context=td_empty._pytree_flatten()[1],
            actual_context=td_populated._pytree_flatten()[1],
            entry_index=1,
            key_path=(),
        )
        assert result == expected

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

        result = diagnose_pytree_structure_mismatch(td_flat, td_nested)
        expected = ContextMismatch(
            expected_context=td_flat._pytree_flatten()[1],
            actual_context=td_nested._pytree_flatten()[1],
            entry_index=1,
            key_path=(),
        )
        assert result == expected

    def test_nested_mismatched_inner_keys_should_fail(self):
        """Nested TensorDicts with mismatched inner keys should fail."""
        from torch.utils._pytree import MappingKey

        td1 = TensorDict(
            {
                "outer": TensorDict({"key1": TENSOR_A, "key2": TENSOR_B}, shape=()),
                "flat": TENSOR_A,
            },
            shape=(),
        )
        td2 = TensorDict(
            {
                "outer": TensorDict({"key1": TENSOR_A, "key3": TENSOR_B}, shape=()),
                "flat": TENSOR_A,
            },
            shape=(),
        )

        result = diagnose_pytree_structure_mismatch(td1, td2)
        expected = ContextMismatch(
            expected_context=td1["outer"]._pytree_flatten()[1],
            actual_context=td2["outer"]._pytree_flatten()[1],
            entry_index=1,
            key_path=(MappingKey("outer"),),
        )
        assert result == expected


class TestListTupleStructures:
    """Test cases for list and tuple PyTree structures."""

    def test_identical_lists(self):
        """Identical list structures should pass."""
        list1 = [TENSOR_A, TENSOR_B]
        list2 = [TENSOR_A, TENSOR_B]
        result = diagnose_pytree_structure_mismatch(list1, list2)
        expected = None
        assert result == expected

    def test_identical_tuples(self):
        """Identical tuple structures should pass."""
        tuple1 = (TENSOR_A, TENSOR_B)
        tuple2 = (TENSOR_A, TENSOR_B)
        result = diagnose_pytree_structure_mismatch(tuple1, tuple2)
        expected = None
        assert result == expected

    def test_list_vs_tuple_should_fail(self):
        """List vs tuple with same content should fail."""
        list_tree = [TENSOR_A, TENSOR_B]
        tuple_tree = (TENSOR_A, TENSOR_B)

        result = diagnose_pytree_structure_mismatch(list_tree, tuple_tree)
        expected = TypeMismatch(
            expected_type=list, actual_type=tuple, entry_index=1, key_path=()
        )
        assert result == expected

    def test_different_nesting_should_fail(self):
        """Trees with different nesting depths should fail."""
        from torch.utils._pytree import SequenceKey

        tree1 = [TENSOR_A, [TENSOR_B]]
        tree2 = [[TENSOR_A], TENSOR_B]

        result = diagnose_pytree_structure_mismatch(tree1, tree2)
        expected = KeyPathMismatch(
            keypaths=((SequenceKey(idx=1),), (SequenceKey(idx=0),))
        )
        assert result == expected


class TestTensorDataClassStructures:
    """Test cases for TensorDataClass PyTree structures."""

    def test_identical_dataclass_structures(self):
        """Identical TensorDataClass structures should pass."""
        dc1 = SimpleDataClass(a=TENSOR_A, b=TENSOR_B, shape=(), device="cpu")
        dc2 = SimpleDataClass(a=TENSOR_A, b=TENSOR_B, shape=(), device="cpu")
        result = diagnose_pytree_structure_mismatch(dc1, dc2)
        expected = None
        assert result == expected

    def test_nested_dataclass_structures(self):
        """Nested TensorDataClass structures with same layout should pass."""
        outer = SimpleDataClass(a=TENSOR_A, b=TENSOR_B, shape=(), device="cpu")

        dc1 = NestedDataClass(outer=outer, flat=TENSOR_A, shape=(), device="cpu")
        dc2 = NestedDataClass(outer=outer, flat=TENSOR_A, shape=(), device="cpu")
        result = diagnose_pytree_structure_mismatch(dc1, dc2)
        expected = None
        assert result == expected

    def test_different_field_names_should_fail(self):
        """TensorDataClasses with different field names should fail."""
        dc1 = SimpleDataClass(a=TENSOR_A, b=TENSOR_B, shape=(), device="cpu")
        dc2 = DifferentFieldDataClass(x=TENSOR_A, y=TENSOR_B, shape=(), device="cpu")

        result = diagnose_pytree_structure_mismatch(dc1, dc2)
        expected = TypeMismatch(
            expected_type=SimpleDataClass,
            actual_type=DifferentFieldDataClass,
            entry_index=1,
            key_path=(),
        )
        assert result == expected

    def test_different_tensor_values_same_structure(self):
        """DataClass instances with different tensor values but same structure should pass."""
        tensor_alt1 = torch.tensor([10, 20])
        tensor_alt2 = torch.tensor([30, 40])

        dc1 = SimpleDataClass(a=TENSOR_A, b=TENSOR_B, shape=(), device="cpu")
        dc2 = SimpleDataClass(a=tensor_alt1, b=tensor_alt2, shape=(), device="cpu")
        result = diagnose_pytree_structure_mismatch(dc1, dc2)
        expected = None
        assert result == expected

    def test_different_batch_shapes_should_fail(self):
        """DataClass instances with different batch shapes should fail."""
        dc1 = SimpleDataClass(a=TENSOR_BATCH, b=TENSOR_BATCH, shape=(2,), device="cpu")
        dc2 = SimpleDataClass(
            a=TENSOR_BATCH, b=TENSOR_BATCH, shape=(2, 3), device="cpu"
        )

        result = diagnose_pytree_structure_mismatch(dc1, dc2)
        expected = ContextMismatch(
            expected_context=dc1._pytree_flatten()[1],
            actual_context=dc2._pytree_flatten()[1],
            entry_index=1,
            key_path=(),
        )
        assert result == expected


class TestMixedStructures:
    """Test cases for mixed PyTree node types."""

    def test_tensordict_vs_list_should_fail(self):
        """TensorDict vs list should fail due to type mismatch."""
        td = TensorDict({"a": TENSOR_A}, shape=())
        list_tree = [TENSOR_A, TENSOR_B]

        result = diagnose_pytree_structure_mismatch(td, list_tree)
        expected = TypeMismatch(
            expected_type=TensorDict, actual_type=list, entry_index=1, key_path=()
        )
        assert result == expected

    def test_list_vs_dict_should_fail(self):
        """List vs dict should fail due to type mismatch."""
        list_tree = [TENSOR_A, TENSOR_B]
        dict_tree = {"0": TENSOR_A, "1": TENSOR_B}

        result = diagnose_pytree_structure_mismatch(list_tree, dict_tree)
        expected = TypeMismatch(
            expected_type=list, actual_type=dict, entry_index=1, key_path=()
        )
        assert result == expected

    def test_tensordict_vs_dataclass_should_fail(self):
        """TensorDict vs TensorDataClass should fail despite similar structure."""
        td = TensorDict({"a": TENSOR_A, "b": TENSOR_B}, shape=())
        dc = SimpleDataClass(a=TENSOR_A, b=TENSOR_B, shape=(), device="cpu")

        result = diagnose_pytree_structure_mismatch(td, dc)
        expected = TypeMismatch(
            expected_type=TensorDict,
            actual_type=SimpleDataClass,
            entry_index=1,
            key_path=(),
        )
        assert result == expected
