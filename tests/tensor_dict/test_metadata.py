import torch
from torch.utils._pytree import tree_map

from tensorcontainer.tensor_dict import TensorDict


class MyObject:
    """A custom object for testing non-standard metadata preservation."""

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, MyObject) and self.value == other.value


class TestBasicMetadata:
    """Tests the preservation of simple, common metadata types."""

    def test_string_metadata_is_preserved(self):
        """
        Verifies that a basic string metadata item is unchanged by pytree operations.
        This serves as a fundamental check for the metadata preservation mechanism.
        """
        td = TensorDict(
            {"tensor": torch.ones(4), "metadata": "some_string"}, shape=(4,)
        )
        td_doubled = tree_map(lambda x: x * 2, td)

        assert torch.allclose(td_doubled["tensor"], torch.ones(4) * 2)
        assert td_doubled["metadata"] == "some_string"


class TestComplexMetadata:
    """Tests more complex metadata types, like mutable objects and custom classes."""

    def test_list_metadata_is_preserved(self):
        """
        Ensures that mutable collection types like lists are correctly preserved.
        """
        td = TensorDict(
            {"tensor": torch.ones(4), "metadata": [1, "a", True]}, shape=(4,)
        )
        td_doubled = tree_map(lambda x: x * 2, td)

        assert torch.allclose(td_doubled["tensor"], torch.ones(4) * 2)
        assert td_doubled["metadata"] == [1, "a", True]

    def test_custom_object_is_preserved(self):
        """
        Ensures that instances of user-defined classes are preserved, which is
        important for storing custom configuration or state.
        """
        custom_obj = MyObject(value=42)
        td = TensorDict({"tensor": torch.ones(4), "metadata": custom_obj}, shape=(4,))
        td_doubled = tree_map(lambda x: x * 2, td)

        assert torch.allclose(td_doubled["tensor"], torch.ones(4) * 2)
        assert td_doubled["metadata"] == custom_obj


class TestStructuralEdgeCases:
    """Tests edge cases related to the structure of the TensorDict."""

    def test_nested_tensordict_with_metadata(self):
        """
        Verifies that metadata is preserved in both parent and nested TensorDicts,
        ensuring the pytree traversal is correct for nested structures.
        """
        td = TensorDict(
            {
                "tensor": torch.ones(4),
                "nested": TensorDict(
                    {"nested_tensor": torch.ones(4, 2), "nested_meta": "level2"},
                    shape=(4,),
                ),
            },
            shape=(4,),
        )
        td_doubled = tree_map(lambda x: x * 2, td)

        assert torch.allclose(td_doubled["tensor"], torch.ones(4) * 2)
        assert torch.allclose(
            td_doubled["nested"]["nested_tensor"], torch.ones(4, 2) * 2
        )
        assert td_doubled["nested"]["nested_meta"] == "level2"

    def test_metadata_only_tensordict(self):
        """
        Tests the edge case where a TensorDict contains no tensors at all, only
        metadata. Pytree operations should not alter it.
        """
        td = TensorDict({"meta1": "a", "meta2": 123}, shape=(4,))
        td_unchanged = tree_map(lambda x: x * 2, td)

        assert td_unchanged.data == td.data

    def test_empty_tensordict(self):
        """
        Tests that an empty TensorDict remains empty and handles pytree
        operations gracefully without errors.
        """
        td = TensorDict({}, shape=(4,))
        td_unchanged = tree_map(lambda x: x * 2, td)

        assert len(td_unchanged) == 0
        assert td_unchanged.shape == (4,)
