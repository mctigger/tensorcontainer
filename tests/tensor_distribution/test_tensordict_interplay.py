import torch
from rtd.tensor_dict import TensorDict
from rtd.tensor_distribution import TensorBernoulli


def test_assign_and_retrieve_tensordistribution():
    # Create an empty TensorDict with scalar batch‐shape
    td = TensorDict({}, shape=(), device=torch.device("cpu"))

    # Create a simple scalar Bernoulli TensorDistribution
    tb = TensorBernoulli(
        probs=torch.tensor(0.7),
        shape=(),
        device=torch.device("cpu"),
        reinterpreted_batch_ndims=0,
    )

    # Assign it
    td["my_dist"] = tb

    # Key must exist and retrieval returns the same object
    assert "my_dist" in td
    assert td["my_dist"] is tb


def test_update_with_tensordistribution():
    # TensorDict with an existing tensor value
    td = TensorDict({"x": torch.ones(2, 2)}, shape=(2,), device=torch.device("cpu"))

    # Two Bernoulli distributions matching the batch‐shape
    tb1 = TensorBernoulli(
        probs=torch.tensor([0.3, 0.6]),
        shape=(2,),
        device=torch.device("cpu"),
        reinterpreted_batch_ndims=0,
    )
    tb2 = TensorBernoulli(
        probs=torch.tensor([0.1, 0.9]),
        shape=(2,),
        device=torch.device("cpu"),
        reinterpreted_batch_ndims=0,
    )

    # Insert via update()
    td.update({"d1": tb1, "d2": tb2})

    # Exact instances must be preserved
    assert td["d1"] is tb1
    assert td["d2"] is tb2


def test_reassign_overwrites_previous_distribution():
    td = TensorDict({"z": torch.randn(3, 3)}, shape=(3,), device=torch.device("cpu"))

    tb_old = TensorBernoulli(
        probs=torch.tensor([0.2, 0.8, 0.5]),
        shape=(3,),
        device=torch.device("cpu"),
        reinterpreted_batch_ndims=0,
    )
    td["dist"] = tb_old

    tb_new = TensorBernoulli(
        probs=torch.tensor([0.6, 0.4, 0.9]),
        shape=(3,),
        device=torch.device("cpu"),
        reinterpreted_batch_ndims=0,
    )
    td["dist"] = tb_new

    # After reassignment, retrieval yields the new object
    assert td["dist"] is tb_new
    assert td["dist"] is not tb_old
