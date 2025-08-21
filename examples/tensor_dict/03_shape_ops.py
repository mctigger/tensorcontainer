"""
TensorDict shape operations.

Key concept: Shape operations (reshape, view, squeeze, unsqueeze) affect only
batch dimensions while automatically preserving each tensor's event dimensions.
"""

import torch
from tensorcontainer import TensorDict


def main() -> None:
    data = TensorDict(
        {
            "observations": torch.randn(2, 3, 128),
            "actions": torch.randn(2, 3, 6),
            "rewards": torch.randn(2, 3),
        },
        shape=(2, 3),
        device="cpu",
    )

    # Reshape only affects batch dimensions
    reshaped = data.reshape(6)
    assert reshaped.shape == (6,)
    assert reshaped["observations"].shape == (6, 128)  # Event dimension preserved

    # Unsqueeze adds batch dimension
    unsqueezed = data.unsqueeze(0)
    assert unsqueezed.shape == (1, 2, 3)

    # Attempt invalid reshape
    try:
        data.reshape(5)  # Cannot reshape (2,3) -> (5,)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
