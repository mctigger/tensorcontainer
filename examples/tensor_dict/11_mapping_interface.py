"""
TensorDict mapping interface.

Key concept: TensorDict implements the MutableMapping interface,
providing dictionary-like methods (keys, values, items, update).
"""

import torch
from tensorcontainer import TensorDict


def main() -> None:
    x = torch.randn(3, 4)
    y = torch.randn(3, 2)
    data = TensorDict({"x": x, "y": y}, shape=(3,), device="cpu")

    # Dictionary-like access methods
    assert data["x"] is x
    assert data["y"] is y

    # Dictionary-like iteration
    for key in data:
        assert key in ["x", "y"]

    for key, value in data.items():
        assert isinstance(value, torch.Tensor)

    # Update from another TensorDict
    z = torch.randn(3, 1)
    other = TensorDict({"z": z}, shape=(3,), device="cpu")
    data.update(other)
    assert data["z"] is z

    # Attempt update with incompatible data
    try:
        data.update({"bad": torch.randn(2, 2)})  # Wrong shape
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
