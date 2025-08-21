"""
TensorDict indexing and slicing.

Key concept: TensorDict supports tensor-like indexing that applies to all
contained tensors simultaneously, preserving batch structure.
"""

import torch
from tensorcontainer import TensorDict


def main() -> None:
    data = TensorDict(
        {
            "x": torch.randn(6, 4),
            "y": torch.randn(6, 2),
        },
        shape=(6,),
        device="cpu",
    )

    # Index a single item - removes batch dimension
    single_item = data[0]
    assert single_item.shape == ()
    assert single_item["x"].shape == (4,)

    # Slice maintains batch structure
    slice_data = data[2:5]
    assert slice_data.shape == (3,)

    # Assignment requires matching shape and keys
    replacement = TensorDict(
        {"x": torch.ones(3, 4), "y": torch.zeros(3, 2)},
        shape=(3,),
        device="cpu",
    )
    data[2:5] = replacement

    # Attempt assignment with mismatched shape
    try:
        wrong_shape = TensorDict(
            {"x": torch.ones(2, 4), "y": torch.zeros(2, 2)},
            shape=(2,),  # Wrong shape - slice expects (3,)
            device="cpu",
        )
        data[2:5] = wrong_shape
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
