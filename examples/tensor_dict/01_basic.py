"""
Basic TensorDict creation and access.

Key concept: TensorDict stores multiple tensors with shared batch dimensions
and allows dictionary-style access to individual tensors.
"""

import torch
from tensorcontainer import TensorDict


def main() -> None:
    # The shape of a TensorDicts must match the leading dimensions of its attributes
    data = TensorDict(
        {
            "observations": torch.randn(4, 10),
            "actions": torch.randn(4, 3),
            "rewards": torch.randn(4),
        },
        shape=(4,),
        device="cpu",
    )

    assert data["observations"].shape == (4, 10)
    assert data.shape == (4,)

    # Attempt to construct with a mismatched shape to demonstrate error handling.
    # The `shape` argument must be a prefix of every field's shape.
    try:
        TensorDict(
            {
                "observations": torch.randn(4, 10),
                "actions": torch.randn(4, 3),
            },
            shape=(3,),  # This shape (3,) is not a prefix of (4, 10) or (4, 3)
            device="cpu",
        )
    except Exception as e:
        # An error is expected here due to the shape mismatch.
        print(e)


if __name__ == "__main__":
    main()