import pytest
import torch

from rtd.tensor_dict import TensorDict  # adjust import as needed
from tests.tensor_dict import common
from tests.tensor_dict.common import compute_cat_shape, compare_nested_dict
from tests.tensor_dict.compile_utils import run_and_compare_compiled

nested_dict = common.nested_dict


# ——— Valid concatenation dims across several shapes ———
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
        ((2, 1, 2), 1),
        ((2, 1, 2), 2),
        ((2, 1, 2), -1),
        ((2, 1, 2), -3),
    ],
)
def test_cat_valid(nested_dict, shape, dim):
    data = nested_dict(shape)
    td = TensorDict(data, shape)

    def cat_fn(td, dim):
        return torch.cat([td, td], dim=dim)

    # concatenate two copies and compare compiled
    run_and_compare_compiled(cat_fn, td, dim)

    # compute expected shape
    expected_shape = compute_cat_shape(shape, dim)
    cat_td = torch.cat([td, td], dim=dim)
    assert cat_td.shape == expected_shape

    compare_nested_dict(data, cat_td, lambda orig: torch.cat([orig, orig], dim=dim))


# ——— Error on invalid dims ———
@pytest.mark.parametrize(
    "shape, dim",
    [
        # 1D: valid dims are [-1..0], so 1 and -2 are invalid
        ((4,), 1),
        ((4,), -2),
        # 2D: valid dims are [-2..1], so 2 and -3 are invalid
        ((2, 2), 2),
        ((2, 2), -3),
        # 3D: valid dims are [-3..2], so 3 and -4 are invalid
        ((2, 1, 2), 3),
        ((2, 1, 2), -4),
    ],
)
def test_cat_invalid_dim_raises(shape, dim, nested_dict):
    td = TensorDict(nested_dict(shape), shape)
    with pytest.raises(IndexError):
        torch.cat([td, td], dim=dim)
