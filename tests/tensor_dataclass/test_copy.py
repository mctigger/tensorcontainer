import torch
import pytest
import copy


@pytest.mark.parametrize("mode", ["eager", "compile"])
def test_tensor_dataclass_shallow_copy(copy_test_instance, mode):
    """Test shallow copy of TensorDataclass with non-tensor metadata."""

    def shallow_copy_logic(data):
        return copy.copy(data)

    if mode == "compile":
        shallow_copy_logic = torch.compile(shallow_copy_logic, fullgraph=True)

    orig = copy_test_instance
    copied = shallow_copy_logic(orig)

    assert orig is not copied
    assert orig.a is copied.a
    assert orig.b is copied.b
    assert orig.metadata is copied.metadata

    orig.metadata.append(4)
    assert copied.metadata == [1, 2, 3, 4]


@pytest.mark.parametrize("mode", ["eager", "compile"])
def test_tensor_dataclass_deep_copy(copy_test_instance, mode):
    """Test deep copy of TensorDataclass with non-tensor metadata."""

    def deep_copy_logic(data):
        return copy.deepcopy(data)

    if mode == "compile":
        deep_copy_logic = torch.compile(deep_copy_logic, fullgraph=True)

    orig = copy_test_instance
    copied = deep_copy_logic(orig)

    assert orig is not copied
    assert orig.a is not copied.a
    assert orig.b is not copied.b
    assert torch.equal(orig.a, copied.a)
    assert torch.equal(orig.b, copied.b)
    assert orig.metadata is not copied.metadata

    orig.metadata.append(4)
    assert copied.metadata == [1, 2, 3]
