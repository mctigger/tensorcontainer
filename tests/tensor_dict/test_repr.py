import torch
from rtd.tensor_dict import TensorDict  # adjust if the file/module is named differently


def test_tensordict_repr():
    td = TensorDict(
        {
            "obs": TensorDict(
                {"image": torch.randn(8, 3, 64, 64), "state": torch.randn(8, 10)},
                shape=(8,),
            ),
            "action": torch.randint(0, 5, (8,)),
        },
        shape=(8,),
        device=torch.device("cpu"),
    )

    repr_str = repr(td)

    # Check top-level structure
    assert "TensorDict" in repr_str
    assert "shape=(8,)" in repr_str
    assert "device=cpu" in repr_str

    # Check keys
    assert "obs:" in repr_str
    assert "action: Tensor(shape=(8,)," in repr_str

    # Check nested keys
    assert "image: Tensor(shape=(8, 3, 64, 64)" in repr_str
    assert "state: Tensor(shape=(8, 10)" in repr_str

    # Check dtypes
    assert "dtype=torch.float32" in repr_str or "dtype=torch.float" in repr_str
    assert "dtype=torch.int64" in repr_str
