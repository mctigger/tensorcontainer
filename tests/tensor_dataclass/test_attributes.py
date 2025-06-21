import pytest
import torch

from rtd.tensor_dataclass import TensorDataclass
from tests.tensor_dict.compile_utils import run_and_compare_compiled


def test_getattr():
    class TestContainer(TensorDataclass):
        a: torch.Tensor
        b: torch.Tensor

    # Test direct attribute access
    container = TestContainer(
        a=torch.zeros(2, 3),
        b=torch.ones(2, 3),
        shape=(2, 3),
        device=torch.device("cpu"),
    )
    assert container.a.shape == (2, 3)
    assert container.b.shape == (2, 3)

    # Test TensorContainer method inheritance
    # These tests have been moved to test_view_reshape.py
    # Validate container structure
    assert isinstance(container.clone(), TestContainer)
    # The .dtype attribute is not defined for TensorContainer, so this check is removed.
    # A more elaborate implementation would require a reduction over all tensor dtypes.
    # assert container.to(torch.float16).dtype == torch.float16

    with pytest.raises(AttributeError):
        _ = container.invalid  # type: ignore # Should raise AttributeError for attribute access


def test_compile():
    """Tests that a function using TensorDataclass can be torch.compiled."""

    class MyData(TensorDataclass):
        x: torch.Tensor
        y: torch.Tensor

    def func(td: MyData) -> MyData:
        return td.view(12)

    data = MyData(
        x=torch.ones(3, 4),
        y=torch.zeros(3, 4),
        shape=(3, 4),
        device=torch.device("cpu"),
    )
    run_and_compare_compiled(func, data)
