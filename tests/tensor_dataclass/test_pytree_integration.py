import torch

from tensorcontainer.tensor_dataclass import TensorDataClass
from tests.stubs import StubTensorContainer


class TestTensorDataClassTensorContainerInteraction:
    def _create_nested_structure(self, shape=(4, 3), requires_grad=False):
        """Helper to create nested TensorDataClass containing StubTensorContainer."""

        class TestTDC(TensorDataClass):
            nested_container: StubTensorContainer
            tensor_a: torch.Tensor

        nested_container = StubTensorContainer(
            tensor_a=torch.randn(*shape, 10, requires_grad=requires_grad),
            tensor_b=torch.randn(*shape, 6, requires_grad=requires_grad),
            shape=shape,
            device=torch.device("cpu"),
        )

        return TestTDC(
            nested_container=nested_container,
            tensor_a=torch.randn(*shape, 5, requires_grad=requires_grad),
            shape=shape,
            device=torch.device("cpu"),
        )

    def test_flatten_nested_containers(self):
        """
        Test that TensorDataClass correctly implements flatten
        when containing nested TensorContainer and verify _pytree_flatten is called.
        """
        tdc = self._create_nested_structure()

        # Use context manager for StubTensorContainer
        with tdc.nested_container:
            # Test PyTree flatten directly
            from torch.utils._pytree import tree_flatten

            flat_tensors, _ = tree_flatten(tdc)

            # Verify StubTensorContainer._pytree_flatten was called
            assert StubTensorContainer._flatten_calls > 0, (
                "StubTensorContainer._pytree_flatten should be called during tree_flatten"
            )

            # Manually verify the flattened structure is correct
            # TensorDataClass flattens recursively, so we get all leaves
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
            assert flat_tensors[0] is tdc.nested_container.tensor_a, (
                "nested tensor_a data mismatch"
            )
            assert flat_tensors[1] is tdc.nested_container.tensor_b, (
                "nested tensor_b data mismatch"
            )
            assert flat_tensors[2] is tdc.tensor_a, "outer tensor_a data mismatch"

    def test_unflatten_nested_containers(self):
        """
        Test that TensorDataClass correctly implements unflatten
        when containing nested TensorContainer and verify _pytree_unflatten is called.
        """
        tdc = self._create_nested_structure()

        # Use context manager for StubTensorContainer
        with tdc.nested_container:
            # Test PyTree flatten/unflatten cycle
            from torch.utils._pytree import tree_flatten, tree_unflatten

            flat_tensors, tree_spec = tree_flatten(tdc)

            # Test reconstruction preserves structure
            reconstructed = tree_unflatten(flat_tensors, tree_spec)

            # Verify StubTensorContainer._pytree_unflatten was called
            assert StubTensorContainer._unflatten_calls > 0, (
                "StubTensorContainer._pytree_unflatten should be called during tree_unflatten"
            )

            assert type(reconstructed) is type(tdc)
            assert reconstructed.shape == tdc.shape
            assert hasattr(reconstructed, "nested_container")
            assert hasattr(reconstructed, "tensor_a")
            assert isinstance(reconstructed.nested_container, StubTensorContainer)

            # Verify reconstructed data matches original
            assert torch.equal(reconstructed.tensor_a, tdc.tensor_a)
            assert torch.equal(
                reconstructed.nested_container.tensor_a, tdc.nested_container.tensor_a
            )
            assert torch.equal(
                reconstructed.nested_container.tensor_b, tdc.nested_container.tensor_b
            )
