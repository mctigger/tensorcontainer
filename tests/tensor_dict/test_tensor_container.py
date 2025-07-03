"""Test the TensorContainer base class methods through TensorDict."""

import pytest
import torch
from torch.utils import _pytree as pytree

from tensorcontainer import TensorDict


class TestTensorContainerTreeMap:
    """
    Tests the `_tree_map` method of the `TensorContainer`.

    This suite verifies that:
    - `_tree_map` behaves like `pytree.tree_map`.
    - `_tree_map` raises an error with a descriptive path.
    """

    def test_tree_map_like_pytree(self):
        """
        Tests that _tree_map works like pytree.tree_map.
        """
        td = TensorDict(
            {
                "a": torch.randn(3, 4),
                "b": {
                    "c": torch.randn(3, 4),
                    "d": torch.randn(3, 4),
                },
            }
        )

        def add_one(x):
            return x + 1

        result_td = td._tree_map(add_one, td)
        expected_td = pytree.tree_map(add_one, td)

        torch.testing.assert_close(result_td["a"], expected_td["a"])
        torch.testing.assert_close(result_td["b"]["c"], expected_td["b"]["c"])
        torch.testing.assert_close(result_td["b"]["d"], expected_td["b"]["d"])

    def test_tree_map_error_path(self):
        """
        Tests that _tree_map raises an error with the correct path.
        """
        td = TensorDict(
            {
                "a": torch.randn(3, 4),
                "b": {
                    "c": "not a tensor",  # This will cause an error
                    "d": torch.randn(3, 4),
                },
            }
        )

        def add_one(x):
            return x + 1

        with pytest.raises(Exception, match=r"Error at path \('b\.c'\)"):
            td._tree_map(add_one, td)
