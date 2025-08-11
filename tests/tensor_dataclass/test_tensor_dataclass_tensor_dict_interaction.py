import torch

from tensorcontainer.tensor_dataclass import TensorDataClass
from tensorcontainer.tensor_dict import TensorDict
from tests.conftest import skipif_no_compile


class TestTensorDataClassTensorDictInteraction:
    def _create_nested_structure(self, shape=(4, 3), requires_grad=False):
        """Helper to create nested TensorDataClass containing TensorDict."""
        
        class TestTDC(TensorDataClass):
            nested_td: TensorDict
            tensor_a: torch.Tensor

        nested_td = TensorDict(
            {
                "tensor_x": torch.randn(*shape, 10, requires_grad=requires_grad),
                "tensor_y": torch.randn(*shape, 6, requires_grad=requires_grad),
            },
            shape=shape,
            device=torch.device("cpu"),
        )
        
        return TestTDC(
            nested_td=nested_td,
            tensor_a=torch.randn(*shape, 5, requires_grad=requires_grad),
            shape=shape,
            device=torch.device("cpu"),
        )

    def test_flatten_nested_containers(self):
        """
        Test that TensorDataClass correctly implements flatten
        when containing nested TensorDict.
        """
        tdc = self._create_nested_structure()
        
        # Test PyTree flatten directly
        from torch.utils._pytree import tree_flatten
        flat_tensors, tree_spec = tree_flatten(tdc)
        
        # Manually verify the flattened structure is correct
        # Expected order: nested_td tensors first (tensor_x, tensor_y), then tensor_a
        assert len(flat_tensors) == 3, f"Expected 3 flattened tensors, got {len(flat_tensors)}"
        
        # Check tensor shapes match expected structure
        assert flat_tensors[0].shape == (4, 3, 10), f"tensor_x shape mismatch: {flat_tensors[0].shape}"
        assert flat_tensors[1].shape == (4, 3, 6), f"tensor_y shape mismatch: {flat_tensors[1].shape}"  
        assert flat_tensors[2].shape == (4, 3, 5), f"tensor_a shape mismatch: {flat_tensors[2].shape}"
        
        # Verify the flattened tensors contain the expected data
        assert torch.equal(flat_tensors[0], tdc.nested_td["tensor_x"]), "tensor_x data mismatch"
        assert torch.equal(flat_tensors[1], tdc.nested_td["tensor_y"]), "tensor_y data mismatch" 
        assert torch.equal(flat_tensors[2], tdc.tensor_a), "tensor_a data mismatch"

    def test_unflatten_nested_containers(self):
        """
        Test that TensorDataClass correctly implements unflatten
        when containing nested TensorDict.
        """
        tdc = self._create_nested_structure()
        
        # Test PyTree flatten/unflatten cycle
        from torch.utils._pytree import tree_flatten, tree_unflatten
        flat_tensors, tree_spec = tree_flatten(tdc)
        
        # Test reconstruction preserves structure
        reconstructed = tree_unflatten(flat_tensors, tree_spec)
        assert type(reconstructed) is type(tdc)
        assert reconstructed.shape == tdc.shape
        assert hasattr(reconstructed, 'nested_td')
        assert hasattr(reconstructed, 'tensor_a')
        assert isinstance(reconstructed.nested_td, TensorDict)
        
        # Verify reconstructed data matches original
        assert torch.equal(reconstructed.tensor_a, tdc.tensor_a)
        assert torch.equal(reconstructed.nested_td["tensor_x"], tdc.nested_td["tensor_x"])
        assert torch.equal(reconstructed.nested_td["tensor_y"], tdc.nested_td["tensor_y"])
        
        # Test that a method relying on _tree_map works correctly
        cloned = tdc.clone()
        assert type(cloned) is type(tdc)
        assert cloned.shape == tdc.shape
        assert isinstance(cloned.nested_td, TensorDict)
        assert torch.equal(cloned.tensor_a, tdc.tensor_a)
        assert torch.equal(cloned.nested_td["tensor_x"], tdc.nested_td["tensor_x"])
