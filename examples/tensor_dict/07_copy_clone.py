"""
TensorDict copy and clone operations.

Key concept: .clone() creates deep copies with independent tensor storage,
while shallow copies share tensor memory with the original.
"""

import torch
import copy
from tensorcontainer import TensorDict


def main() -> None:
    original = TensorDict(
        {"x": torch.randn(3, 4), "y": torch.randn(3, 2)},
        shape=(3,),
        device="cpu",
    )

    # Shallow copy shares tensor memory
    shallow_copy = copy.copy(original)
    assert shallow_copy["x"] is original["x"] # Same tensor objects
    assert shallow_copy["y"] is original["y"]

    # Clone creates independent tensors
    cloned = original.clone()
    assert cloned["x"] is not original["x"] # Different tensor objects
    assert cloned["y"] is not original["y"]


if __name__ == "__main__":
    main()