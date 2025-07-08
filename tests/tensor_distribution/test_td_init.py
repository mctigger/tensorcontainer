import torch

from src.tensorcontainer.tensor_distribution.normal import TensorNormal
from tests.conftest import skipif_no_compile


class TestTensorDistributionInit:
    def test_eager_init(self):
        # Test initialization in eager mode
        loc = torch.randn(2, 3)
        scale = torch.rand(2, 3)
        td = TensorNormal(
            loc=loc,
            scale=scale,
            reinterpreted_batch_ndims=1,
            shape=loc.shape,
            device=loc.device,
        )
        assert torch.equal(td.loc, loc)
        assert torch.equal(td.scale, scale)

    @skipif_no_compile
    def test_compile_init(self):
        """
        Verifies that a TensorNormal can be successfully created from raw tensors
        within a torch.compile'd function.
        """

        def create_td_from_tensors(loc_arg, scale_arg):
            td = TensorNormal(
                loc=loc_arg,
                scale=scale_arg,
                reinterpreted_batch_ndims=1,
                shape=loc_arg.shape,
                device=loc_arg.device,
            )
            return td.loc, td.scale

        loc = torch.randn(2, 3)
        scale = torch.rand(2, 3)
        from tests.compile_utils import run_and_compare_compiled

        eager_result, compiled_result = run_and_compare_compiled(
            create_td_from_tensors, loc, scale
        )
        assert torch.allclose(eager_result[0], compiled_result[0])
        assert torch.allclose(eager_result[1], compiled_result[1])
