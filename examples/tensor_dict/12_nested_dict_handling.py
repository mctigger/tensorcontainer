"""
Automatic nested dictionary handling.

Key concept: TensorDict automatically converts nested plain dictionaries
into TensorDict instances during construction.
"""

import torch
from tensorcontainer import TensorDict


def main() -> None:
    # Plain dicts are automatically wrapped as TensorDict instances
    data = TensorDict(
        {
            "env": {
                "observations": torch.randn(3, 10),
                "info": {
                    "step_count": torch.tensor([10, 15, 20]),
                }
            },
            "agent": {
                "policy": torch.randn(3, 6),
            },
            "reward": torch.randn(3),
        },
        shape=(3,),
        device="cpu",
    )

    # Verify automatic wrapping
    assert isinstance(data["env"], TensorDict)
    assert isinstance(data["env"]["info"], TensorDict)
    assert isinstance(data["agent"], TensorDict)
    assert isinstance(data["reward"], torch.Tensor)

    # Access nested data
    assert data["env"]["observations"].shape == (3, 10)
    assert data["env"]["info"]["step_count"].shape == (3,)
    
    # Operations propagate through automatic nesting
    reshaped = data.reshape(1, 3)
    assert reshaped["env"]["info"]["step_count"].shape == (1, 3)

    # Attempt nested dict with wrong shape
    try:
        TensorDict(
            {
                "valid": {"tensor": torch.randn(3, 2)},
                "invalid": {"tensor": torch.randn(2, 2)},  # Wrong batch size
            },
            shape=(3,),
            device="cpu",
        )
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()