import pytest
import torch

from rtd.tensor_dict import TensorDict
from tests.tensor_dict.compile_utils import run_and_compare_compiled
from tests.tensor_dict import common

nested_dict = common.nested_dict


# ——— Valid stacking dims across several shapes ———


@pytest.mark.parametrize(
    "shape, dim",
    [
        # 1D
        ((4,), 0),
        ((4,), -1),
        # 2D
        ((2, 2), 0),
        ((2, 2), 1),
        ((2, 2), -1),
        ((1, 4), 0),
        ((1, 4), 1),
        ((1, 4), -2),
        # 3D
        ((2, 1, 2), 0),
        ((2, 1, 2), 3),
        ((2, 1, 2), -1),
        ((2, 1, 2), -4),
    ],
)
def test_stack_valid_compiled(nested_dict, shape, dim):
    def stack_fn(nd, s, d):
        td = TensorDict(nd(s), s)
        return torch.stack([td, td], dim=d)

    run_and_compare_compiled(stack_fn, nested_dict, shape, dim)


# ——— Error on invalid dims ———
@pytest.mark.parametrize(
    "shape, dim",
    [
        # 1D: valid dims are in [-2..1], so 2 and -3 are out of range
        ((4,), 2),
        ((4,), -3),
        # 2D: valid dims are in [-3..2], so 3 and -4 are out of range
        ((2, 2), 3),
        ((2, 2), -4),
        # 3D: valid dims are in [-4..3], so 4 and -5 are out of range
        ((2, 1, 2), 4),
        ((2, 1, 2), -5),
    ],
)
def test_stack_invalid_dim_raises(shape, dim, nested_dict):
    td = TensorDict(nested_dict(shape), shape)
    with pytest.raises(IndexError):
        torch.stack([td, td], dim=dim)
