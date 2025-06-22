import pytest
import torch
from torch._dynamo import exc as dynamo_exc

from rtd.tensor_dataclass import TensorDataclass
from tests.conftest import skipif_no_compile
from tests.tensor_dataclass.conftest import (
    compute_cat_shape,
    SimpleTensorData,
    SHAPE_DIM_PARAMS_VALID,
    SHAPE_DIM_PARAMS_INVALID,
    create_inconsistent_meta_data_pair,
)
from tests.tensor_dict.compile_utils import run_and_compare_compiled


class TestCat:
    """Test suite for torch.cat operations on TensorDataclass instances."""

    @staticmethod
    def _cat_operation(tensor_dict_instance_list, dim_arg):
        """Helper method for cat operations."""
        return torch.cat(tensor_dict_instance_list, dim=dim_arg)

    def _create_tensor_data_pair(self, shape, device="cpu"):
        """Helper method to create a pair of SimpleTensorData instances."""
        device_obj = torch.device(device)
        td1 = SimpleTensorData(
            a=torch.randn(shape, device=device),
            b=torch.ones(shape, device=device),
            shape=shape,
            device=device_obj,
        )
        td2 = SimpleTensorData(
            a=torch.randn(shape, device=device),
            b=torch.ones(shape, device=device),
            shape=shape,
            device=device_obj,
        )
        return td1, td2

    def _verify_cat_result(self, cat_td, td1, td2, shape, dim):
        """Helper method to verify cat operation results."""
        expected_shape = compute_cat_shape(shape, dim)
        assert cat_td.shape == expected_shape
        assert cat_td.a.shape == expected_shape
        assert cat_td.b.shape == expected_shape

        # Calculate actual dimension and split point
        actual_dim = dim if dim >= 0 else dim + len(shape)
        split_point = shape[actual_dim]

        # Create slices for verification
        slices_td1_list = [slice(None)] * len(shape)
        slices_td2_list = [slice(None)] * len(shape)
        slices_td1_list[actual_dim] = slice(0, split_point)
        slices_td2_list[actual_dim] = slice(split_point, None)

        slices_td1 = tuple(slices_td1_list)
        slices_td2 = tuple(slices_td2_list)

        # Verify tensor values
        assert torch.equal(cat_td.a[slices_td1], td1.a)
        assert torch.equal(cat_td.a[slices_td2], td2.a)
        assert torch.equal(cat_td.b[slices_td1], td1.b)
        assert torch.equal(cat_td.b[slices_td2], td2.b)

    @pytest.mark.parametrize("shape, dim", SHAPE_DIM_PARAMS_VALID)
    def test_cat_valid(self, shape, dim):
        """Test cat operation with valid dimensions."""
        td1, td2 = self._create_tensor_data_pair(shape)
        cat_td = self._cat_operation([td1, td2], dim)
        self._verify_cat_result(cat_td, td1, td2, shape, dim)

    @pytest.mark.parametrize("shape, dim", SHAPE_DIM_PARAMS_INVALID)
    @pytest.mark.parametrize("compile_mode", [False, True])
    def test_cat_invalid_dim_raises(self, shape, dim, compile_mode):
        """Test cat operation raises with invalid dimensions in both eager and compile modes."""
        if compile_mode:
            pytest.importorskip("torch._dynamo", reason="Compilation not available")

        td1, td2 = self._create_tensor_data_pair(shape)

        if compile_mode:
            compiled_cat_op = torch.compile(self._cat_operation, fullgraph=True)
            with pytest.raises(dynamo_exc.TorchRuntimeError) as excinfo:
                compiled_cat_op([td1, td2], dim)
            assert "IndexError" in str(
                excinfo.value
            ) and "Dimension out of range" in str(excinfo.value)
        else:
            with pytest.raises(IndexError):
                self._cat_operation([td1, td2], dim)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("compile_mode", [False, True])
    def test_cat_on_cuda(self, compile_mode):
        """Test cat operation on CUDA devices in both eager and compile modes."""
        if compile_mode:
            pytest.importorskip("torch._dynamo", reason="Compilation not available")

        td1, td2 = self._create_tensor_data_pair((2, 3), device="cuda")

        if compile_mode:
            run_and_compare_compiled(self._cat_operation, [td1, td2], 0)
        else:
            cat_td = self._cat_operation([td1, td2], 0)
            assert cat_td.device.type == "cuda"
            assert cat_td.a.device.type == "cuda"  # type: ignore
            assert cat_td.b.device.type == "cuda"  # type: ignore
            assert cat_td.shape == (4, 3)

    @pytest.mark.parametrize("compile_mode", [False, True])
    def test_cat_inconsistent_meta_data_raises(self, compile_mode):
        """Test cat operation raises with inconsistent metadata in both eager and compile modes."""
        if compile_mode:
            pytest.importorskip("torch._dynamo", reason="Compilation not available")

        class MetaData(TensorDataclass):
            a: torch.Tensor
            b: torch.Tensor
            meta: int = 42

        td1, td2 = create_inconsistent_meta_data_pair(
            MetaData, (2, 3), torch.device("cpu"), [42, 99]
        )

        if compile_mode:
            with pytest.raises(dynamo_exc.Unsupported):
                torch.compile(self._cat_operation, fullgraph=True)([td1, td2], 0)
        else:
            with pytest.raises(ValueError):
                self._cat_operation([td1, td2], 0)

    @pytest.mark.parametrize("compile_mode", [False, True])
    def test_cat_inconsistent_shapes_raises(
        self, simple_tensor_data_instance, compile_mode
    ):
        """Test cat operation raises with inconsistent shapes in both eager and compile modes."""
        if compile_mode:
            pytest.importorskip("torch._dynamo", reason="Compilation not available")

        td1 = simple_tensor_data_instance
        td2 = SimpleTensorData(
            a=torch.randn(2, 4),  # Inconsistent shape
            b=torch.ones(2, 4),
            shape=(2, 4),
            device=torch.device("cpu"),
        )

        if compile_mode:
            with pytest.raises(dynamo_exc.Unsupported):
                torch.compile(self._cat_operation, fullgraph=True)([td1, td2], 0)
        else:
            with pytest.raises(ValueError):
                self._cat_operation([td1, td2], 0)

    @skipif_no_compile
    def test_cat_compile(self, simple_tensor_data_instance):
        """Test cat operation compilation."""

        def func(tds, dim_arg):
            return torch.cat(tds, dim=dim_arg)

        td1 = simple_tensor_data_instance
        td2 = SimpleTensorData(
            a=td1.a * 2,
            b=td1.b * 2,
            shape=td1.shape,
            device=td1.device,
        )
        run_and_compare_compiled(func, [td1, td2], 0)

    def test_cat_empty_list_raises(self):
        """Test cat operation raises with empty list."""
        with pytest.raises(RuntimeError, match="expected a non-empty list of Tensors"):
            torch.cat([], dim=0)

    def test_cat_inconsistent_non_cat_dim_raises(self, simple_tensor_data_instance):
        """Test cat operation raises with inconsistent non-cat dimensions."""
        td1 = simple_tensor_data_instance
        td2 = SimpleTensorData(
            shape=(2, 4),  # Inconsistent dimension 1
            device=torch.device("cpu"),
            a=torch.randn(2, 4),
            b=torch.randn(2, 4),
        )

        with pytest.raises(ValueError, match="Node context mismatch"):
            torch.cat([td1, td2], dim=0)  # type: ignore
