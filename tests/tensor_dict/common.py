import pytest
import torch


@pytest.fixture
def nested_dict():
    def _make(shape):
        nested_dict_data = {  # Renamed to avoid conflict with outer scope
            "x": {
                "a": torch.arange(0, 4).reshape(*shape),
                "b": torch.arange(4, 8).reshape(*shape),
            },
            "y": torch.arange(8, 12).reshape(*shape),
        }
        return nested_dict_data

    return _make


def compute_cat_shape(shape, dim):
    ndim = len(shape)
    # normalize negative dim
    pos_dim = dim if dim >= 0 else dim + ndim
    expected_shape = list(shape)
    expected_shape[pos_dim] = expected_shape[pos_dim] * 2
    return tuple(expected_shape)


def compute_stack_shape(shape, dim, num_tensors=2):
    ndim = len(shape)
    # For stack, dim can be from -(ndim + 1) to ndim.
    # If dim is -(ndim + 1), it's equivalent to 0.
    # If dim is ndim, it's equivalent to ndim.
    # Other negative dims: dim + (ndim + 1)
    if dim < 0:
        pos_dim = dim + ndim + 1
    else:
        pos_dim = dim

    expected_shape = list(shape)
    expected_shape.insert(pos_dim, num_tensors)
    return tuple(expected_shape)


def compare_nested_dict(data, output, expect_fn):
    for key, val in data.items():
        if isinstance(val, dict):
            for subkey, orig in val.items():
                out = output[key][subkey]
                expect = expect_fn(orig)
                assert out.shape == expect.shape, (
                    f"Shape mismatch for {key}.{subkey}: {out.shape} vs {expect.shape}"
                )
                assert torch.equal(out, expect), f"Tensor mismatch for {key}.{subkey}"
        else:
            orig = val
            out = output[key]
            expect = expect_fn(orig)
            assert out.shape == expect.shape, (
                f"Shape mismatch for {key}: {out.shape} vs {expect.shape}"
            )
            assert torch.equal(out, expect), f"Tensor mismatch for {key}"
