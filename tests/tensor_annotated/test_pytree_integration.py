import torch

from tensorcontainer import TensorAnnotated
from tests.stubs import StubTensorContainer


class TestTensorAnnotatedTensorContainerInteraction:
    def _create_nested_structure(self, shape=(4, 3), requires_grad=False):
        """Helper to create nested TensorAnnotated containing StubTensorContainer."""

        class TestTA(TensorAnnotated):
            nested_container: StubTensorContainer
            tensor_a: torch.Tensor

            def __init__(
                self,
                nested_container: StubTensorContainer,
                tensor_a: torch.Tensor,
                **kwargs,
            ):
                self.nested_container = nested_container
                self.tensor_a = tensor_a
                init_shape = kwargs.pop(
                    "shape", tensor_a.shape[:-1]
                )  # Remove last dimension for batch shape
                init_device = kwargs.pop("device", tensor_a.device)
                super().__init__(shape=init_shape, device=init_device)

        # Create StubTensorContainer directly
        nested_container = StubTensorContainer(
            tensor_a=torch.randn(*shape, 10, requires_grad=requires_grad),
            tensor_b=torch.randn(*shape, 6, requires_grad=requires_grad),
            shape=shape,
            device=torch.device("cpu"),
        )

        return TestTA(
            nested_container=nested_container,
            tensor_a=torch.randn(*shape, 5, requires_grad=requires_grad),
            shape=shape,
            device=torch.device("cpu"),
        )

    def test_flatten_nested_containers(self):
        """
        Test that TensorAnnotated correctly implements flatten
        when containing nested TensorContainer and verify _pytree_flatten is called.
        """
        ta = self._create_nested_structure()
        from torch.utils._pytree import tree_flatten

        with ta.nested_container:
            flat_tensors, _ = tree_flatten(ta)
            assert StubTensorContainer._flatten_calls > 0, (
                "StubTensorContainer._pytree_flatten should be called during tree_flatten"
            )
            assert len(flat_tensors) == 3, (
                f"Expected 3 flattened tensors, got {len(flat_tensors)}"
            )
            assert flat_tensors[0].shape == (4, 3, 10), (
                f"nested tensor_a shape mismatch: {flat_tensors[0].shape}"
            )
            assert flat_tensors[1].shape == (4, 3, 6), (
                f"nested tensor_b shape mismatch: {flat_tensors[1].shape}"
            )
            assert flat_tensors[2].shape == (4, 3, 5), (
                f"outer tensor_a shape mismatch: {flat_tensors[2].shape}"
            )
            assert flat_tensors[0] is ta.nested_container.tensor_a, (
                "nested tensor_a data mismatch"
            )
            assert flat_tensors[1] is ta.nested_container.tensor_b, (
                "nested tensor_b data mismatch"
            )
            assert flat_tensors[2] is ta.tensor_a, "outer tensor_a data mismatch"

    def test_unflatten_nested_containers(self):
        """
        Test that TensorAnnotated correctly implements unflatten
        when containing nested TensorContainer and verify _pytree_unflatten is called.
        """
        ta = self._create_nested_structure()
        from torch.utils._pytree import tree_flatten, tree_unflatten

        with ta.nested_container:
            flat_tensors, tree_spec = tree_flatten(ta)
            reconstructed = tree_unflatten(flat_tensors, tree_spec)

            assert StubTensorContainer._unflatten_calls > 0, (
                "StubTensorContainer._pytree_unflatten should be called during tree_unflatten"
            )
            assert isinstance(reconstructed, TensorAnnotated)
            assert reconstructed.shape == ta.shape
            assert hasattr(reconstructed, "nested_container")
            assert hasattr(reconstructed, "tensor_a")
            assert isinstance(reconstructed.nested_container, StubTensorContainer)
            assert torch.equal(reconstructed.tensor_a, ta.tensor_a)
            assert torch.equal(
                reconstructed.nested_container.tensor_a, ta.nested_container.tensor_a
            )
            assert torch.equal(
                reconstructed.nested_container.tensor_b, ta.nested_container.tensor_b
            )
