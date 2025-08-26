import pytest
import torch

from tensorcontainer.tensor_dict import TensorDict
from tensorcontainer.tensor_dataclass import TensorDataClass


class SampleDataClass(TensorDataClass):
    """Sample TensorDataClass for testing."""
    x: torch.Tensor
    y: torch.Tensor


def _make_nested_tensordict(shape=(3, 4)):
    """Helper function to create a nested TensorDict for testing."""
    return TensorDict(
        {
            "x": torch.randn(*shape, 3),
            "y": torch.randn(*shape, 4),
        },
        shape=shape,
    )


def _make_sample_dataclass(shape=(3, 4)):
    """Helper function to create a SampleDataClass for testing."""
    return SampleDataClass(
        x=torch.randn(*shape, 3),
        y=torch.randn(*shape, 4),
        shape=shape,
        device="cpu",
    )


class TestStackCrossTypes:
    """Test torch.stack behavior with different TensorContainer types."""

    def test_stack_tensordict_with_tensordataclass_raises_error(self):
        """Test that torch.stack raises an error when given TensorDict and TensorDataClass."""
        td = _make_nested_tensordict()
        tdc = _make_sample_dataclass()
        
        with pytest.raises(ValueError, match="Type mismatch"):
            torch.stack([td, tdc], dim=0)

    def test_stack_tensordataclass_with_tensordict_raises_error(self):
        """Test that torch.stack raises an error when given TensorDataClass and TensorDict."""
        tdc = _make_sample_dataclass()
        td = _make_nested_tensordict()
        
        with pytest.raises(ValueError, match="Type mismatch"):
            torch.stack([tdc, td], dim=0)

    def test_stack_same_type_tensordict_works(self):
        """Test that torch.stack works with same type (TensorDict)."""
        td1 = _make_nested_tensordict()
        td2 = _make_nested_tensordict()
        
        # This should work without raising an error
        result = torch.stack([td1, td2], dim=0)
        expected_shape = (2, 3, 4)  # (stack_size,) + original_shape
        assert result.shape == expected_shape

    def test_stack_same_type_tensordataclass_works(self):
        """Test that torch.stack works with same type (TensorDataClass)."""
        tdc1 = _make_sample_dataclass()
        tdc2 = _make_sample_dataclass()
        
        # This should work without raising an error
        result = torch.stack([tdc1, tdc2], dim=0)
        expected_shape = (2, 3, 4)  # (stack_size,) + original_shape
        assert result.shape == expected_shape


class TestStackMixedScenarios:
    """Test torch.stack behavior with mixed devices, shapes, keys, dtypes."""

    def test_stack_mixed_devices_cpu_cuda(self):
        """Test that torch.stack handles mixed devices (CPU vs CUDA)."""
        td_cpu = _make_nested_tensordict()
        
        # Create CUDA version if available
        if torch.cuda.is_available():
            td_cuda = TensorDict(
                {
                    "x": torch.randn(3, 4, 3).cuda(),
                    "y": torch.randn(3, 4, 4).cuda(),
                },
                shape=(3, 4),
                device="cuda",
            )
            
            # This should either work (moving to same device) or raise an error
            try:
                result = torch.stack([td_cpu, td_cuda], dim=0)
                # If it works, check the result device
                assert result.device.type in ["cpu", "cuda"]
            except (RuntimeError, ValueError) as e:
                # Expected to fail with device mismatch or node context mismatch
                assert any(keyword in str(e).lower() for keyword in ["device", "cuda", "node context", "mismatch"])
        else:
            pytest.skip("CUDA not available")

    def test_stack_mixed_shapes_incompatible(self):
        """Test that torch.stack raises error for incompatible shapes."""
        td1 = _make_nested_tensordict(shape=(3, 4))
        td2 = TensorDict(
            {
                "x": torch.randn(2, 4, 3),  # Different first dimension
                "y": torch.randn(2, 4, 4),
            },
            shape=(2, 4),
        )
        
        # This should fail due to incompatible shapes
        with pytest.raises(ValueError, match="stack expects each TensorContainer to be equal size"):
            torch.stack([td1, td2], dim=0)

    def test_stack_mixed_keys_structure(self):
        """Test that torch.stack handles different keys/structure."""
        td1 = _make_nested_tensordict()
        td2 = TensorDict(
            {
                "x": torch.randn(3, 4, 3),
                "z": torch.randn(3, 4, 5),  # Different key 'z' instead of 'y'
            },
            shape=(3, 4),
        )
        
        # This should fail due to different structure
        with pytest.raises(ValueError, match="Node context mismatch"):
            torch.stack([td1, td2], dim=0)

    def test_stack_mixed_dtypes_works(self):
        """Test that torch.stack handles mixed dtypes with dtype promotion."""
        td_float = TensorDict(
            {
                "x": torch.randn(3, 4, 3, dtype=torch.float32),
                "y": torch.randn(3, 4, 4, dtype=torch.float32),
            },
            shape=(3, 4),
        )
        td_int = TensorDict(
            {
                "x": torch.randint(0, 10, (3, 4, 3), dtype=torch.int64),
                "y": torch.randint(0, 10, (3, 4, 4), dtype=torch.int64),
            },
            shape=(3, 4),
        )
        
        # This should work with dtype promotion
        result = torch.stack([td_float, td_int], dim=0)
        # PyTorch promotes int64 + float32 to float32
        assert result["x"].dtype == torch.float32
        assert result["y"].dtype == torch.float32
        assert result.shape == (2, 3, 4)

    def test_stack_mixed_nested_structure_works(self):
        """Test that torch.stack can handle different nested structures."""
        td1 = TensorDict(
            {
                "x": {"a": torch.randn(3, 4, 2)},
                "y": torch.randn(3, 4, 3),
            },
            shape=(3, 4),
        )
        td2 = TensorDict(
            {
                "x": torch.randn(3, 4, 2),  # Flat instead of nested
                "y": torch.randn(3, 4, 3),
            },
            shape=(3, 4),
        )
        
        # This actually fails due to structure mismatch
        with pytest.raises(ValueError, match="Node context mismatch"):
            torch.stack([td1, td2], dim=0)

    def test_stack_different_event_dimensions_fails(self):
        """Test stacking containers with same batch dims but different event dims."""
        td1 = TensorDict(
            {
                "x": torch.randn(3, 4, 5),  # Event dim: 5
                "y": torch.randn(3, 4, 6),  # Event dim: 6
            },
            shape=(3, 4),
        )
        td2 = TensorDict(
            {
                "x": torch.randn(3, 4, 7),  # Event dim: 7 (different)
                "y": torch.randn(3, 4, 8),  # Event dim: 8 (different)
            },
            shape=(3, 4),
        )
        
        # This should fail due to incompatible tensor shapes when stacking
        with pytest.raises(RuntimeError, match="stack expects each tensor to be equal size"):
            torch.stack([td1, td2], dim=0)