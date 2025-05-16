from rtd.tensor_dict import TensorDict
import torch
import pytest


def test_some_function():
    td = TensorDict({"a": torch.tensor([[1.0]])}, shape=[1, 1])

    def fn(td: TensorDict):
        x = td["a"]  # direct access
        y = x.view(-1)  # apply view before returning
        return {"a": y}

    compiled_fn = torch.compile(fn)
    out = compiled_fn(td)

    assert isinstance(out, dict)
    assert "a" in out
    assert out["a"].shape == torch.Size([1])
    assert torch.allclose(out["a"], td["a"].view(-1))


@pytest.fixture
def td():
    return TensorDict(
        {"a": torch.arange(6).reshape(3, 2), "b": torch.ones(3, 2)}, shape=[3]
    )


def test_index_int(td):
    sub = td[1]
    assert isinstance(sub, TensorDict)
    assert sub["a"].shape == (2,)
    assert torch.equal(sub["a"], td["a"][1])


def test_index_slice(td):
    sub = td[1:]
    assert isinstance(sub, TensorDict)
    assert sub["a"].shape == (2, 2)
    assert torch.equal(sub["a"], td["a"][1:])


def test_index_tensor(td):
    idx = torch.tensor([0, 2])
    sub = td[idx]
    assert isinstance(sub, TensorDict)
    assert sub["a"].shape == (2, 2)
    assert torch.equal(sub["a"], td["a"][idx])


def test_index_boolean_mask(td):
    mask = torch.tensor([True, False, True])
    sub = td[mask]
    assert isinstance(sub, TensorDict)
    assert sub["a"].shape == (2, 2)
    assert torch.equal(sub["a"], td["a"][mask])


def test_index_none(td):
    sub = td[None]
    assert isinstance(sub, TensorDict)
    assert sub["a"].shape == (1, 3, 2)
    assert torch.equal(sub["a"], td["a"].unsqueeze(0))


def test_index_ellipsis(td):
    sub = td[...]
    assert isinstance(sub, TensorDict)
    assert torch.equal(sub["a"], td["a"][...])  # full copy


def test_index_tuple(td):
    sub = td[1, ...]
    assert isinstance(sub, TensorDict)
    assert torch.equal(sub["a"], td["a"][1, ...])


def test_nested_dict_indexing():
    td = TensorDict(
        {
            "x": TensorDict({"y": torch.arange(4).reshape(2, 2)}, shape=[2]),
            "z": torch.ones(2, 2),
        },
        shape=[2],
    )

    sub = td[1]
    assert isinstance(sub, TensorDict)
    assert isinstance(sub["x"], TensorDict)
    assert sub["x"]["y"].shape == (2,)
    assert torch.equal(sub["x"]["y"], torch.tensor([2, 3]))
