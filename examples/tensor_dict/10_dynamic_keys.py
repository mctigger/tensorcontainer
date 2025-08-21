"""
Dynamic key management with TensorDict.

Key concept: TensorDict allows runtime addition and removal of keys,
unlike TensorDataClass which has a static schema.
"""

import torch
from tensorcontainer import TensorDict


def main() -> None:
    data = TensorDict(
        {"observations": torch.randn(4, 10)},
        shape=(4,),
        device="cpu",
    )

    # Add keys dynamically
    data["actions"] = torch.randn(4, 3)
    data["rewards"] = torch.randn(4)

    # Remove keys dynamically
    del data["rewards"]
    assert "rewards" not in data

    # Attribute shapes must be compatible with TensorDict shape
    try:
        data["wrong_shape"] = torch.randn(
            3, 2
        )  # Wrong batch shape (3,) instead of (4,)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
