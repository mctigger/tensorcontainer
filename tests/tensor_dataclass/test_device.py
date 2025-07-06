import pytest
import torch

from tensorcontainer.tensor_dataclass import TensorDataClass
from tests.conftest import skipif_no_compile
from tests.tensor_dataclass.conftest import assert_device_consistency


class TestDevice:
    """Test class for device-related functionality."""

    @pytest.mark.parametrize("compile_mode", [False, True])
    def test_to_device(self, device_test_instance, compile_mode):
        """Test moving tensor dataclass to different device."""
        if compile_mode:
            pytest.importorskip("torch", minversion="2.0")
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")

        td = device_test_instance.to(torch.device("cuda"))
        assert_device_consistency(td, torch.device("cuda"))

    def test_device_consistency_check(self, device_test_instance):
        """Test that device consistency validation catches mismatches."""
        from tests.tensor_dataclass.conftest import DeviceTestClass

        with pytest.raises(RuntimeError):
            DeviceTestClass(
                a=torch.randn(2, 3, device=torch.device("cuda")),
                b=torch.ones(2, 3),
                shape=(2, 3),
                device=torch.device("cpu"),
            )

    @skipif_no_compile
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_compile(self, device_test_instance):
        """Test device operations work with torch.compile."""

        def device_fn(td):
            return td.to("cuda")

        compiled_fn = torch.compile(device_fn, fullgraph=True)
        result = compiled_fn(device_test_instance)

        assert_device_consistency(result, torch.device("cuda"))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_nested_device_mismatch_raises(self):
        """Test that device validation catches mismatches in nested TensorDataclasses."""

        class Inner(TensorDataClass):
            c: torch.Tensor

        class Outer(TensorDataClass):
            inner: Inner

        with pytest.raises(RuntimeError):
            Outer(
                shape=(2,),
                device=torch.device("cpu"),
                inner=Inner(
                    shape=(2,),
                    device=torch.device("cuda"),
                    c=torch.randn(2, device="cuda"),
                ),
            )
