import pytest
import torch

from .conftest import assert_device_consistency


class TestTo:
    """Test .to() method functionality of TensorDataclass."""

    def test_to_different_device(self, to_test_instance):
        """Test moving TensorDataclass to a different device."""
        td = to_test_instance

        # Move to CUDA if available
        if torch.cuda.is_available():
            td_cuda = td.to(torch.device("cuda"))
            assert_device_consistency(td_cuda, torch.device("cuda"))
        else:
            # Move to a different CPU device
            td_cpu1 = td.to(torch.device("cpu"))
            assert_device_consistency(td_cpu1, torch.device("cpu"))

    def test_to_same_device(self, to_test_instance):
        """Test moving TensorDataclass to the same device."""
        td = to_test_instance

        # Move to the same device
        td_same = td.to(torch.device("cpu"))
        assert_device_consistency(td_same, torch.device("cpu"))

    def test_to_with_dtype_change(self, to_test_instance):
        """Test moving TensorDataclass with dtype change."""
        td = to_test_instance

        # Move to float64
        td_double = td.to(dtype=torch.float64)

        assert td_double.a.dtype == torch.float64
        assert td_double.b.dtype == torch.float64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_with_non_blocking_and_memory_format(self, to_test_4d_instance):
        """Test moving TensorDataclass with non_blocking and memory_format arguments."""
        td = to_test_4d_instance

        # Move with non_blocking=True and channels_last memory format
        td_non_blocking = td.to(
            torch.device("cuda"), non_blocking=True, memory_format=torch.channels_last
        )

        assert_device_consistency(td_non_blocking, torch.device("cuda"))
        # Check if the tensor is in channels_last format by verifying its layout
        assert td_non_blocking.a.is_contiguous(memory_format=torch.channels_last)
        assert td_non_blocking.b.is_contiguous(memory_format=torch.channels_last)

    def test_to_mixed_fields(self, to_test_instance):
        """Test moving a TensorDataclass with mixed tensor and non-tensor fields."""
        td = to_test_instance

        # Move to CUDA if available
        if torch.cuda.is_available():
            td_cuda = td.to(torch.device("cuda"))
            assert_device_consistency(td_cuda, torch.device("cuda"))
            assert td_cuda.meta == 42  # Non-tensor field should remain unchanged
        else:
            # Move to a different CPU device
            td_cpu1 = td.to(torch.device("cpu"))
            assert_device_consistency(td_cpu1, torch.device("cpu"))
            assert td_cpu1.meta == 42  # Non-tensor field should remain unchanged

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_device_inference(self, to_test_instance):
        """Test that the device attribute is correctly inferred after .to() calls."""
        td = to_test_instance

        # Move to "cuda" and check if the device attribute is correctly inferred
        td_cuda = td.to("cuda")
        assert td_cuda.device == torch.device("cuda:0")

        # Move to a specific CUDA device
        td_cuda_1 = td.to("cuda:0")
        assert td_cuda_1.device == torch.device("cuda:0")

        # Test with a dtype change only, device should not change
        td_float64 = td.to(dtype=torch.float64)
        assert td_float64.device == td.device
