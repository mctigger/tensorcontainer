import torch
from tensorcontainer import TensorDataClass


class DataPair(TensorDataClass):
    x: torch.Tensor
    y: torch.Tensor


def main():
    # Create a small batch of tensors. The leading dimension is the batch (B).
    # Indexing behaves like with torch.Tensor and returns a TensorDataClass.
    x = torch.arange(3 * 4).reshape(3, 4)  # [B=3, F=4]
    y = torch.arange(3)  # [B=3]
    data = DataPair(x=x, y=y, shape=(3,), device="cpu")

    # 1) Tensor-like indexing
    # Integer indexing selects a single item; slicing selects a sub-batch.
    first = data[0]  # single item (no batch dims)
    tail = data[1:]  # sub-batch with B=2
    print("Indexing returns TensorDataClass with the expected shapes:")
    print(
        f"data.shape={tuple(data.shape)}, first.shape={tuple(first.shape)}, tail.shape={tuple(tail.shape)}"
    )
    print(f"tail.x.shape={tuple(tail.x.shape)}, tail.y.shape={tuple(tail.y.shape)}")

    # 2) Views: indexing returns a new DataPair whose tensors are views of the same storage.
    # Modifying the tensors inside the indexed view modifies the original data.
    print("\nIndexing produces views (modifying the view modifies the source):")
    before = int(data.x[0, 0])
    first.x[0] = -1  # in-place write through the view
    after = int(data.x[0, 0])
    print(f"x[0,0] before={before} -> after={after}")

    # 3) Set with another TensorDataClass of compatible shape.
    # You can assign to an indexed TensorDataClass using another instance with matching batch shape.
    print("\nAssign to a slice with another TensorDataClass:")
    replacement = DataPair(
        x=torch.full((2, 4), 99),
        y=torch.tensor([1000, 1001]),
        shape=(2,),
        device="cpu",
    )
    data[1:3] = replacement
    print(
        f"after assignment: x[1,0]={int(data.x[1, 0])}, x[2,0]={int(data.x[2, 0])}, y[1]={int(data.y[1])}, y[2]={int(data.y[2])}"
    )


if __name__ == "__main__":
    main()
