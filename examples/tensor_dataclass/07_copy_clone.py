"""Cloning a TensorDataClass to create independent tensor storage.

This example demonstrates how to use clone() to create a new TensorDataClass
instance with detached tensor storage but identical values. The cloned instance
is completely independent - mutations to its tensors do not affect the original.
Note: copy() provides a deepcopy-like alternative but is not shown here.
"""

import torch
from tensorcontainer import TensorDataClass


# Step 1: Define a simple TensorDataClass with two tensor fields
class DataPair(TensorDataClass):
    """A simple data container with two tensor fields."""

    x: torch.Tensor  # First tensor field
    y: torch.Tensor  # Second tensor field


def main():
    # Step 2: Create deterministic tensors for our example
    x_tensor = torch.tensor([1, 2, 3])  # Shape: (3,)
    y_tensor = torch.tensor([10, 20, 30])  # Shape: (3,)

    # Step 3: Instantiate the original TensorDataClass
    original = DataPair(
        x=x_tensor,
        y=y_tensor,
        shape=(3,),  # Batch size of 3
        device="cpu",
    )

    # Step 4: Clone the original to create an independent copy
    cloned = original.clone()

    # Step 5: Print both instances to show they have identical values initially
    print("Before mutation:")
    print(f"original.x = {original.x}")
    # original.x = tensor([1, 2, 3])
    print(f"cloned.x = {cloned.x}")
    # cloned.x = tensor([1, 2, 3])

    # Step 6: Mutate the cloned instance to demonstrate independence
    cloned.x[0] = -1  # Change first element of cloned.x

    # Step 7: Print both instances to show the original is unaffected
    print("\nAfter mutating cloned.x[0] = -1:")
    print(f"original.x = {original.x}")
    # original.x = tensor([1, 2, 3])
    print(f"cloned.x = {cloned.x}")
    # cloned.x = tensor([-1,  2,  3])


if __name__ == "__main__":
    main()
