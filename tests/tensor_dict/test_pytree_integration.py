import torch

from tensorcontainer.tensor_dict import TensorDict
from tests.stubs import StubTensorContainer


class TestTensorDictTensorContainerInteraction:
    def _create_nested_structure(self, shape=(4, 3), requires_grad=False):
        """Helper to create nested TensorDict containing StubTensorContainer."""

        nested_container = StubTensorContainer(
            tensor_a=torch.randn(*shape, 10, requires_grad=requires_grad),
            tensor_b=torch.randn(*shape, 6, requires_grad=requires_grad),
            shape=shape,
            device=torch.device("cpu"),
        )

        return TensorDict(
            {
                "nested_container": nested_container,
                "tensor_a": torch.randn(*shape, 5, requires_grad=requires_grad),
            },
            shape=shape,
            device=torch.device("cpu"),
        )

    def test_flatten_nested_containers(self):
        """
        Test that TensorDict correctly implements flatten
        when containing nested TensorContainer and verify _pytree_flatten is called.
        """
        td = self._create_nested_structure()

        # Use context manager for StubTensorContainer
        with td["nested_container"]:
            # Test PyTree flatten directly
            from torch.utils._pytree import tree_flatten

            flat_tensors, _ = tree_flatten(td)

            # Verify StubTensorContainer._pytree_flatten was called
            assert StubTensorContainer._flatten_calls > 0, (
                "StubTensorContainer._pytree_flatten should be called during tree_flatten"
            )

            # Manually verify the flattened structure is correct
            # TensorDict flattens to its immediate values, which includes the StubTensorContainer
            # The StubTensorContainer then gets flattened recursively
            # Expected: StubTensorContainer's tensor_a, tensor_b, then outer tensor_a
            assert len(flat_tensors) == 3, (
                f"Expected 3 flattened tensors, got {len(flat_tensors)}"
            )

            # Check tensor shapes match expected structure
            assert flat_tensors[0].shape == (4, 3, 10), (
                f"nested tensor_a shape mismatch: {flat_tensors[0].shape}"
            )
            assert flat_tensors[1].shape == (4, 3, 6), (
                f"nested tensor_b shape mismatch: {flat_tensors[1].shape}"
            )
            assert flat_tensors[2].shape == (4, 3, 5), (
                f"outer tensor_a shape mismatch: {flat_tensors[2].shape}"
            )

            # Verify the flattened tensors contain the expected data
            assert flat_tensors[0] is td["nested_container"].tensor_a, (
                "nested tensor_a data mismatch"
            )
            assert flat_tensors[1] is td["nested_container"].tensor_b, (
                "nested tensor_b data mismatch"
            )
            assert flat_tensors[2] is td["tensor_a"], "outer tensor_a data mismatch"

    def test_unflatten_nested_containers(self):
        """
        Test that TensorDict correctly implements unflatten
        when containing nested TensorContainer and verify _pytree_unflatten is called.
        """
        td = self._create_nested_structure()

        # Use context manager for StubTensorContainer
        with td["nested_container"]:
            # Test PyTree flatten/unflatten cycle
            from torch.utils._pytree import tree_flatten, tree_unflatten

            flat_tensors, tree_spec = tree_flatten(td)

            # Test reconstruction preserves structure
            reconstructed = tree_unflatten(flat_tensors, tree_spec)

            # Verify StubTensorContainer._pytree_unflatten was called
            assert StubTensorContainer._unflatten_calls > 0, (
                "StubTensorContainer._pytree_unflatten should be called during tree_unflatten"
            )

            assert type(reconstructed) is TensorDict
            assert reconstructed.shape == td.shape
            assert "nested_container" in reconstructed
            assert "tensor_a" in reconstructed
            assert isinstance(reconstructed["nested_container"], StubTensorContainer)

            # Verify reconstructed data matches original
            assert reconstructed["tensor_a"] is td["tensor_a"]
            assert (
                reconstructed["nested_container"].tensor_a
                is td["nested_container"].tensor_a
            )
            assert (
                reconstructed["nested_container"].tensor_b
                is td["nested_container"].tensor_b
            )
