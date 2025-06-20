import pytest
import torch


@pytest.fixture
def nested_dict():
    def _make(shape):
        nested_dict_data = {
            "x": {
                "a": torch.arange(0, 4).reshape(*shape),
                "b": torch.arange(4, 8).reshape(*shape),
            },
            "y": torch.arange(8, 12).reshape(*shape),
        }
        return nested_dict_data

    return _make
