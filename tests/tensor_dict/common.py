import pytest
import torch


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


def compute_cat_shape(shape, dim):
    ndim = len(shape)
    # normalize negative dim
    pos_dim = dim if dim >= 0 else dim + ndim
    expected_shape = list(shape)
    expected_shape[pos_dim] = expected_shape[pos_dim] * 2
    return tuple(expected_shape)


def compare_nested_dict(data, output, expect_fn):
    for key, val in data.items():
        if isinstance(val, dict):
            for subkey, orig in val.items():
                out = output[key][subkey]
                expect = expect_fn(orig)
                assert out.shape == expect.shape
                assert torch.equal(out, expect)
        else:
            orig = val
            out = output[key]
            expect = expect_fn(orig)
            assert out.shape == expect.shape
            assert torch.equal(out, expect)
