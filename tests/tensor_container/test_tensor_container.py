import pytest
import torch
import torch.utils._pytree as pytree
from torch.utils._pytree import tree_map

from src.tensorcontainer.tensor_dict import TensorDict


def _make_nested_tensordict(shape=(2,)):
    return TensorDict(
        {
            "a": torch.randn(*shape, 3),
            "b": TensorDict(
                {
                    "c": torch.randn(*shape, 4, 5),
                    "d": torch.randn(*shape, 6),
                },
                shape=shape,
            ),
            "e": torch.randn(*shape),
        },
        shape=shape,
    )


class TestTreeMap:
    def test_tree_map_behaves_like_pytree_tree_map(self):
        td = _make_nested_tensordict()

        def func(x):
            return x * 2

        # Apply using _tree_map from TensorDict (which inherits from TensorContainer)
        result_td = td._tree_map(func)

        # Apply using direct pytree.tree_map on the internal data
        expected_data = tree_map(func, td.data)
        expected_td = TensorDict(expected_data, shape=td.shape, device=td.device)

        # Compare the results by flattening and comparing leaves
        result_leaves = pytree.tree_leaves(result_td)
        expected_leaves = pytree.tree_leaves(expected_td)
        for res_leaf, exp_leaf in zip(result_leaves, expected_leaves):
            assert torch.allclose(res_leaf, exp_leaf)

    def test_tree_map_exception_handling(self):
        td = _make_nested_tensordict()

        # Function that raises an exception for a specific path
        def func_with_error(x):
            if x is td["b"]["c"]:
                raise ValueError("Simulated error at 'b.c'")

            return x  # Ensure a return for non-error paths

        with pytest.raises(Exception) as excinfo:
            td._tree_map(func_with_error)

        assert "Error at path ['b']['c']" in str(excinfo.value)
