import torch
from tensorcontainer import TensorDataClass


class SimpleData(TensorDataClass):
    observations: torch.Tensor
    actions: torch.Tensor


def main() -> None:
    # What this example shows:
    # - The constructor is auto-generated from the annotated fields (like Python dataclasses).
    # - Unlike plain dataclasses, you must also pass `shape` and `device` when constructing.
    #   `shape` gives the leading batch dimensions shared by all fields and MUST be a prefix
    #   of each tensor's shape.
    # - You can access fields via dot notation.

    # Two toy tensors with a shared leading batch of size 2.
    obs_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    act_tensor = torch.tensor([0, 1])  # (2,)

    # Construction: same feel as dataclasses, but we additionally pass `shape` and `device`.
    data = SimpleData(
        observations=obs_tensor,
        actions=act_tensor,
        shape=(2,),  # batch prefix shared by all fields
        device="cpu",
    )

    # Concise output: repr + a quick dot-notation access.
    print("Instance:", data)  # dataclass-style repr
    print("First action via dot notation:", data.actions[0].item())

    # Shape must be a prefix of every field. This mismatched shape will raise.
    try:
        _ = SimpleData(
            observations=obs_tensor,
            actions=act_tensor,
            shape=(3,),  # not a prefix of (2, 3) or (2,)
            device="cpu",
        )
    except Exception as err:  # noqa: BLE001
        print(f"Shape mismatch detected: {type(err).__name__}")


if __name__ == "__main__":
    main()
