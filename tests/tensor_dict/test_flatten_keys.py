import pytest
import torch

from tensorcontainer.tensor_dict import TensorDict
from tests.compile_utils import run_and_compare_compiled


@pytest.fixture
def simple_nested():
    """A TensorDict with one level of nesting."""
    data = {
        "x": {
            "a": torch.tensor([[1, 2], [3, 4]]),
            "b": torch.tensor([[5, 6], [7, 8]]),
        },
        "y": torch.tensor([[9, 10], [11, 12]]),
    }
    return TensorDict(data, shape=(2, 2))


@pytest.fixture
def deep_nested():
    """A deeper nested structure."""
    data = {
        "a": {"b": {"c": torch.arange(4).reshape(2, 2)}},
        "d": {"e": torch.ones(2, 2)},
    }
    return TensorDict(data, shape=(2, 2))


def test_flatten_keys_simple(simple_nested):
    td = simple_nested
    td_flat = td.flatten_keys()
    run_and_compare_compiled(td.flatten_keys)
    assert "x.a" in td_flat.keys()
    assert "x.b" in td_flat.keys()
    assert "y" in td_flat.keys()
    assert isinstance(list(td_flat.keys())[0], str)


def test_flatten_keys_custom_sep(simple_nested):
    td = simple_nested

    def flatten_keys_custom_sep(td):
        return td.flatten_keys(separator="_")

    td_flat = flatten_keys_custom_sep(td)
    run_and_compare_compiled(flatten_keys_custom_sep, td)
    assert "x_a" in td_flat.keys()
    assert "x_b" in td_flat.keys()
    assert "y" in td_flat.keys()
    assert isinstance(list(td_flat.keys())[0], str)


def test_flatten_keys_deep(deep_nested):
    td = deep_nested
    td_flat = td.flatten_keys()
    run_and_compare_compiled(td.flatten_keys)
    assert "a.b.c" in td_flat.keys()
    assert "d.e" in td_flat.keys()
    assert isinstance(list(td_flat.keys())[0], str)


def test_flatten_keys_idempotent_on_flat():
    """Applying flatten_keys twice should yield the same result as applying it once."""
    data = {"z": torch.zeros(3, 4)}
    td = TensorDict(data, shape=(3, 4))

    def flatten_keys(td):
        return td.flatten_keys()

    td_flat = flatten_keys(td)
    run_and_compare_compiled(flatten_keys, td)
    assert "z" in td_flat.keys()
    assert isinstance(list(td_flat.keys())[0], str)


def test_flatten_keys_empty():
    td = TensorDict({}, shape=())
    td_flat = td.flatten_keys()
    run_and_compare_compiled(td.flatten_keys)
    assert len(td_flat.keys()) == 0


def test_flatten_keys_complex():
    """A more complex nested structure."""
    data = {
        "a": {
            "b": {
                "c": torch.arange(4).reshape(2, 2),
                "d": torch.ones(2, 2),
            },
            "e": torch.zeros(2, 2),
        },
        "f": torch.full((2, 2), 2),
    }
    td = TensorDict(data, shape=(2, 2))
    td_flat = td.flatten_keys()
    run_and_compare_compiled(td.flatten_keys)
    assert "a.b.c" in td_flat.keys()
    assert "a.b.d" in td_flat.keys()
    assert "a.e" in td_flat.keys()
    assert "f" in td_flat.keys()
    assert isinstance(list(td_flat.keys())[0], str)


def test_flatten_keys_torch_compile(simple_nested, deep_nested):
    td_simple = simple_nested
    td_simple_flat = td_simple.flatten_keys()
    run_and_compare_compiled(td_simple.flatten_keys)
    assert "x.a" in td_simple_flat.keys()
    assert "x.b" in td_simple_flat.keys()
    assert "y" in td_simple_flat.keys()
    assert isinstance(list(td_simple_flat.keys())[0], str)

    td_deep = deep_nested

    def flatten_func(d):
        return d.flatten_keys()

    td_deep_flat = flatten_func(td_deep)
    run_and_compare_compiled(flatten_func, td_deep)
    assert "a.b.c" in td_deep_flat.keys()
    assert "d.e" in td_deep_flat.keys()
    assert isinstance(list(td_deep_flat.keys())[0], str)

    def flatten_custom_sep(td):
        return td.flatten_keys(separator="_")

    td_simple = simple_nested
    td_custom_flat = flatten_custom_sep(td_simple)
    run_and_compare_compiled(flatten_custom_sep, td_simple)
    assert "x_a" in td_custom_flat.keys()
    assert "x_b" in td_custom_flat.keys()
    assert "y" in td_custom_flat.keys()
    assert isinstance(list(td_custom_flat.keys())[0], str)

    td_empty = TensorDict({}, shape=())
    td_empty_flat = td_empty.flatten_keys()
    run_and_compare_compiled(td_empty.flatten_keys)
    assert len(td_empty_flat.keys()) == 0


def test_flatten_keys_items_correctly_mapped(simple_nested):
    """Uses .items() to check that values are in the correct keys after flattening."""
    td_original = simple_nested
    td_flat = td_original.flatten_keys()

    # Expected key-value pairs after flattening
    expected_data = {
        "x.a": td_original["x"]["a"],
        "x.b": td_original["x"]["b"],
        "y": td_original["y"],
    }

    # Check that the number of items in the flattened TensorDict is as expected.
    assert len(td_flat.keys()) == len(expected_data)

    # Iterate over the .items() of the flattened TensorDict
    for key, value in td_flat.items():
        # Check that the flattened key is one of the expected keys
        assert key in expected_data

        # Check that the tensor value is correct
        torch.testing.assert_close(value, expected_data[key])

        # Check that it's the same tensor object, not a copy.
        # This is an important property of flatten_keys.
        assert value is expected_data[key]


def test_flatten_keys_getitem_access(simple_nested):
    """Checks that accessing values by key on a flattened TensorDict works correctly."""
    td_original = simple_nested
    td_flat = td_original.flatten_keys()

    # Get original values for comparison
    original_value_xa = td_original["x"]["a"]
    original_value_xb = td_original["x"]["b"]
    original_value_y = td_original["y"]

    # Access values using __getitem__ (i.e., td[key]) and assert correctness
    torch.testing.assert_close(td_flat["x.a"], original_value_xa)
    torch.testing.assert_close(td_flat["x.b"], original_value_xb)
    torch.testing.assert_close(td_flat["y"], original_value_y)

    # Also assert that the retrieved values are the same tensor objects, not copies
    assert td_flat["x.a"] is original_value_xa
    assert td_flat["x.b"] is original_value_xb
    assert td_flat["y"] is original_value_y
