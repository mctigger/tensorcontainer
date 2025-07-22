import pytest
import torch

from tensorcontainer.tensor_dataclass import TensorDataClass
from tensorcontainer.tensor_dict import TensorDict
from tests.conftest import skipif_no_compile, skipif_no_cuda
from tests.tensor_dataclass.conftest import assert_nested_device_consistency


class StandardDataClass(TensorDataClass):
    a: torch.Tensor
    b: torch.Tensor
    meta: int = 42


class TestToDevice:
    """Test .to() method for device operations."""

    def test_to_same_device(self):
        """Test moving to the same device."""
        td = StandardDataClass(
            a=torch.randn(2, 3),
            b=torch.ones(2, 3),
            shape=(2, 3),
            device=torch.device("cpu"),
        )

        result = td.to("cpu")
        assert result.a.device == torch.device("cpu")
        assert result.b.device == torch.device("cpu")

    @skipif_no_cuda
    def test_to_cuda(self):
        """Test moving to CUDA device."""
        td = StandardDataClass(
            a=torch.randn(2, 3),
            b=torch.ones(2, 3),
            shape=(2, 3),
            device=torch.device("cpu"),
        )

        result = td.to("cuda")
        assert result.a.device == torch.device("cuda:0")
        assert result.b.device == torch.device("cuda:0")
        assert result.device == torch.device("cuda:0")

    @skipif_no_cuda
    def test_to_specific_cuda_device(self):
        """Test moving to a specific CUDA device."""
        td = StandardDataClass(
            a=torch.randn(2, 3),
            b=torch.ones(2, 3),
            shape=(2, 3),
            device=torch.device("cpu"),
        )

        result = td.to("cuda:0")
        assert result.a.device == torch.device("cuda:0")
        assert result.b.device == torch.device("cuda:0")
        assert result.device == torch.device("cuda:0")

    @pytest.mark.parametrize("compile_mode", [False, True])
    @skipif_no_cuda
    def test_to_device_with_torch_compile(self, device_test_instance, compile_mode):
        """Test moving tensor dataclass to different device with optional torch.compile."""
        if compile_mode:
            pytest.importorskip("torch", minversion="2.0")

        td = device_test_instance.to(torch.device("cuda"))
        assert_nested_device_consistency(td, torch.device("cuda"))

    @skipif_no_compile
    @skipif_no_cuda
    def test_to_device_compile(self, device_test_instance):
        """Test device operations work with torch.compile."""

        def device_fn(td):
            return td.to("cuda")

        compiled_fn = torch.compile(device_fn, fullgraph=True)
        result = compiled_fn(device_test_instance)

        assert_nested_device_consistency(result, torch.device("cuda"))


class TestToDtype:
    """Test .to() method for dtype operations."""

    def test_to_dtype_change(self):
        """Test changing tensor dtype."""
        td = StandardDataClass(
            a=torch.randn(2, 3),
            b=torch.ones(2, 3),
            shape=(2, 3),
            device=torch.device("cpu"),
        )

        result = td.to(dtype=torch.float64)
        assert result.a.dtype == torch.float64
        assert result.b.dtype == torch.float64

    def test_to_dtype_preserves_device(self):
        """Test that dtype change preserves device."""
        td = StandardDataClass(
            a=torch.randn(2, 3),
            b=torch.ones(2, 3),
            shape=(2, 3),
            device=torch.device("cpu"),
        )

        result = td.to(dtype=torch.float64)
        assert result.device == td.device

    def test_to_int_dtype(self):
        """Test changing to integer dtype."""
        td = StandardDataClass(
            a=torch.randn(2, 3),
            b=torch.ones(2, 3),
            shape=(2, 3),
            device=torch.device("cpu"),
        )

        result = td.to(dtype=torch.int32)
        assert result.a.dtype == torch.int32
        assert result.b.dtype == torch.int32


class TestToMemoryFormat:
    """Test .to() method for memory format operations."""

    @skipif_no_cuda
    def test_to_with_memory_format(self):
        """Test moving with memory format specification."""
        td = StandardDataClass(
            a=torch.randn(2, 3, 4, 5),
            b=torch.ones(2, 3, 4, 5),
            shape=(2, 3, 4, 5),
            device=torch.device("cpu"),
        )

        result = td.to("cuda", memory_format=torch.channels_last)
        assert result.a.device == torch.device("cuda:0")
        assert result.b.device == torch.device("cuda:0")
        assert result.a.is_contiguous(memory_format=torch.channels_last)
        assert result.b.is_contiguous(memory_format=torch.channels_last)

    @skipif_no_cuda
    def test_to_with_non_blocking(self):
        """Test non-blocking transfer."""
        td = StandardDataClass(
            a=torch.randn(2, 3, 4, 5),
            b=torch.ones(2, 3, 4, 5),
            shape=(2, 3, 4, 5),
            device=torch.device("cpu"),
        )

        result = td.to("cuda", non_blocking=True)
        assert result.a.device == torch.device("cuda:0")
        assert result.b.device == torch.device("cuda:0")


class TestToPreservation:
    """Test .to() method for field preservation."""

    @skipif_no_cuda
    def test_preserves_non_tensor_fields(self):
        """Test that non-tensor fields are preserved."""
        td = StandardDataClass(
            a=torch.randn(2, 3),
            b=torch.ones(2, 3),
            shape=(2, 3),
            device=torch.device("cpu"),
        )

        result = td.to("cuda")
        assert result.meta == 42


class TestToNested:
    """Test .to() method for nested container operations."""

    @skipif_no_cuda
    def test_nested_containers(self):
        """Test that device changes propagate to all nested TensorContainer subclasses."""

        # Create nested TensorDataClass
        class Inner(TensorDataClass):
            inner_tensor: torch.Tensor

        class Outer(TensorDataClass):
            nested_dataclass: Inner
            nested_tensordict: TensorDict
            direct_tensor: torch.Tensor

        # Create nested TensorDict
        nested_td = TensorDict(
            {
                "a": torch.randn(2, 3),
                "b": torch.ones(2, 4),
            },
            shape=(2,),
            device=torch.device("cpu"),
        )

        # Create nested TensorDataClass
        inner = Inner(
            inner_tensor=torch.randn(2, 5),
            shape=(2,),
            device=torch.device("cpu"),
        )

        # Create outer container
        outer = Outer(
            nested_dataclass=inner,
            nested_tensordict=nested_td,
            direct_tensor=torch.randn(2, 6),
            shape=(2,),
            device=torch.device("cpu"),
        )

        # Move to CUDA
        cuda_outer = outer.to("cuda")

        # Verify device propagation to all levels using helper function
        assert_nested_device_consistency(cuda_outer, torch.device("cuda"))
