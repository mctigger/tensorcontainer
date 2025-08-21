"""
TensorDict stacking and concatenation.

Key concept: torch.stack and torch.cat work with TensorDict instances,
applying the operation to all contained tensors simultaneously.
"""

import torch
from tensorcontainer import TensorDict


def main() -> None:
    batch1 = TensorDict(
        {"x": torch.randn(3, 4), "y": torch.randn(3, 2)}, shape=(3,), device="cpu"
    )

    batch2 = TensorDict(
        {"x": torch.randn(3, 4), "y": torch.randn(3, 2)}, shape=(3,), device="cpu"
    )

    # Stack creates new batch dimension
    stacked = torch.stack([batch1, batch2], dim=0)
    assert stacked.shape == (2, 3)
    assert stacked["x"].shape == (2, 3, 4)

    # Concatenate along existing dimension
    concatenated = torch.cat([batch1, batch2], dim=0)
    assert concatenated.shape == (6,)

    # Attempt to stack with mismatched keys
    try:
        batch3 = TensorDict({"x": torch.randn(3, 4)}, shape=(3,), device="cpu")
        torch.stack([batch1, batch3], dim=0)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
