import torch

from tensorcontainer.tensor_dict import TensorDict
from tests.compile_utils import run_and_compare_compiled


class TestTensorDictDetach:
    def test_detach_returns_new_tensordict(self):
        """Test that detach() creates a new TensorDict instance."""
        # Create TensorDict with tensors that don't require gradients
        td = TensorDict(
            {
                "observations": torch.randn(4, 10),
                "actions": torch.randn(4, 5),
            },
            shape=(4,),
        )

        detached_td = td.detach()

        # Should return a new TensorDict instance
        assert isinstance(detached_td, TensorDict)
        assert detached_td is not td

    def test_detached_tensors_lose_gradients(self):
        """Test that detach() removes gradient tracking from all tensors."""
        # Create TensorDict with tensors that require gradients
        td = TensorDict(
            {
                "weights": torch.randn(3, 8, requires_grad=True),
                "biases": torch.randn(3, 1, requires_grad=True),
            },
            shape=(3,),
        )

        detached_td = td.detach()

        # All detached tensors should not require gradients
        assert not detached_td["weights"].requires_grad
        assert not detached_td["biases"].requires_grad

    def test_original_tensordict_unchanged_after_detach(self):
        """Test that the original TensorDict is not modified by detach()."""
        # Create TensorDict with gradient-tracking tensors
        td = TensorDict(
            {
                "params": torch.randn(2, 6, requires_grad=True),
                "state": torch.randn(2, 4, requires_grad=True),
            },
            shape=(2,),
        )

        # Call detach but ignore the result
        td.detach()

        # Original tensors should still require gradients
        assert td["params"].requires_grad
        assert td["state"].requires_grad

    def test_detached_tensors_share_memory_storage(self):
        """Test that detach() creates tensors sharing the same memory storage."""
        # Create TensorDict with gradient-tracking tensors
        td = TensorDict(
            {
                "data": torch.randn(5, 3, requires_grad=True),
                "labels": torch.randn(5, 1, requires_grad=True),
            },
            shape=(5,),
        )

        detached_td = td.detach()

        # Detached tensors should share memory with original tensors
        assert td["data"].data_ptr() == detached_td["data"].data_ptr()
        assert td["labels"].data_ptr() == detached_td["labels"].data_ptr()

    def test_detach_works_with_torch_compile(self):
        """Test that detach() is compatible with torch.compile."""
        # Create TensorDict with gradient-tracking tensors
        td = TensorDict(
            {
                "input": torch.randn(3, 7, requires_grad=True),
                "target": torch.randn(3, 2, requires_grad=True),
            },
            shape=(3,),
        )

        def detach_operation(tensor_dict):
            return tensor_dict.detach()

        # Compare eager and compiled execution
        eager_result, compiled_result = run_and_compare_compiled(detach_operation, td)

        # Compiled result should be a proper TensorDict
        assert isinstance(compiled_result, TensorDict)
        assert compiled_result is not td

        # Compiled result should not have gradient tracking
        assert not compiled_result["input"].requires_grad
        assert not compiled_result["target"].requires_grad

        # Original should still have gradient tracking
        assert td["input"].requires_grad
        assert td["target"].requires_grad

        # Should share memory storage
        assert td["input"].data_ptr() == compiled_result["input"].data_ptr()
        assert td["target"].data_ptr() == compiled_result["target"].data_ptr()
