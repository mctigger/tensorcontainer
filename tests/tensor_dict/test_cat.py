import pytest
import torch

from rtd.tensor_dict import TensorDict  # adjust import as needed


@pytest.fixture
def nested_dict():
    def _make(shape):
        nested_dict = {
            "x": {
                "a": torch.arange(0, 4).reshape(*shape),
                "b": torch.arange(4, 8).reshape(*shape),
            },
            "y": torch.arange(8, 12).reshape(*shape),
        }
        return nested_dict

    return _make


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
    print("TENSORDICT", td)

    # concatenate two copies
    cat_td = torch.cat([td, td], dim=dim)
    print(cat_td)

    # compute expected shape
    ndim = len(shape)
    # normalize negative dim
    pos_dim = dim if dim >= 0 else dim + ndim
    expected_shape = list(shape)
    expected_shape[pos_dim] = expected_shape[pos_dim] * 2
    assert cat_td.shape == tuple(expected_shape)

    # compare every leaf
    for key, val in data.items():
        if isinstance(val, dict):
            for subkey, orig in val.items():
                out = cat_td[key][subkey]
                expect = torch.cat([orig, orig], dim=dim)
                assert out.shape == expect.shape
                assert torch.equal(out, expect)
        else:
            orig = val
            out = cat_td[key]
            expect = torch.cat([orig, orig], dim=dim)
            assert out.shape == expect.shape
            assert torch.equal(out, expect)


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
