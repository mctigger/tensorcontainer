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
        ((2, 2), 2),
        ((2, 2), -1),
        ((1, 4), 0),
        ((1, 4), 1),
        ((1, 4), 2),
        ((1, 4), -2),
        # 3D
        ((2, 1, 2), 0),
        ((2, 1, 2), 3),
        ((2, 1, 2), -1),
        ((2, 1, 2), -4),
    ],
)
def test_stack_valid(nested_dict, shape, dim):
    # build the TensorDict
    data = nested_dict(shape)
    td = TensorDict(data, shape)

    # stack two copies
    stacked = torch.stack([td, td], dim=dim)

    # compute the expected shape after inserting a new axis of size 2
    orig_ndim = len(shape)
    new_ndim = orig_ndim + 1
    # normalize negative dims
    pos_dim = dim if dim >= 0 else dim + new_ndim
    expected_shape = list(shape)
    expected_shape.insert(pos_dim, 2)
    assert stacked.shape == tuple(expected_shape)

    # every leaf tensor should match torch.stack of the underlying tensors
    for key, val in data.items():
        if isinstance(val, dict):
            for subkey, orig in val.items():
                out = stacked[key][subkey]
                expect = torch.stack([orig, orig], dim=dim)
                assert out.shape == expect.shape
                assert torch.equal(out, expect)
        else:
            orig = val
            out = stacked[key]
            expect = torch.stack([orig, orig], dim=dim)
            assert out.shape == expect.shape
            assert torch.equal(out, expect)


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
