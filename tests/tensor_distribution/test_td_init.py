import torch
from src.rtd.tensor_distribution import TensorNormal


class TestTensorDistributionInit:
    def test_eager_init(self):
        # Test initialization in eager mode
        loc = torch.randn(2, 3)
        scale = torch.rand(2, 3)
        td = TensorNormal(loc, scale, reinterpreted_batch_ndims=1)
        assert torch.equal(td["loc"], loc)
        assert torch.equal(td["scale"], scale)

    def test_compile_init(self):
        """
        Verifies that a TensorNormal can be successfully created from raw tensors
        within a torch.compile'd function.
        """

        def create_td_from_tensors(loc_arg, scale_arg):
            td = TensorNormal(loc_arg, scale_arg, reinterpreted_batch_ndims=1)
            return td["loc"], td["scale"]

        loc = torch.randn(2, 3)
        scale = torch.rand(2, 3)
        from tests.tensor_dict.compile_utils import run_and_compare_compiled

        eager_result, compiled_result = run_and_compare_compiled(
            create_td_from_tensors, loc, scale
        )
        assert torch.allclose(eager_result[0], compiled_result[0])
        assert torch.allclose(eager_result[1], compiled_result[1])
