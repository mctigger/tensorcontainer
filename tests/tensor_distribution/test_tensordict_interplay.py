import torch

from tensorcontainer.tensor_dict import TensorDict
from tensorcontainer.tensor_distribution.bernoulli import TensorBernoulli


def test_assign_and_retrieve_tensordistribution():
    # Create an empty TensorDict with scalar batch‐shape
    td = TensorDict({}, shape=(), device=torch.device("cpu"))

    # Create a simple scalar Bernoulli TensorDistribution
    probs = torch.tensor(0.7)
    tb = TensorBernoulli(
        _probs=probs,
        reinterpreted_batch_ndims=0,
        shape=probs.shape,
        device=probs.device,
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
    probs1 = torch.tensor([0.3, 0.6])
    tb1 = TensorBernoulli(
        _probs=probs1,
        reinterpreted_batch_ndims=0,
        shape=probs1.shape,
        device=probs1.device,
    )
    probs2 = torch.tensor([0.1, 0.9])
    tb2 = TensorBernoulli(
        _probs=probs2,
        reinterpreted_batch_ndims=0,
        shape=probs2.shape,
        device=probs2.device,
    )

    # Insert via update()
    td.update({"d1": tb1, "d2": tb2})

    # Exact instances must be preserved
    assert td["d1"] is tb1
    assert td["d2"] is tb2


def test_reassign_overwrites_previous_distribution():
    td = TensorDict({"z": torch.randn(3, 3)}, shape=(3,), device=torch.device("cpu"))

    probs_old = torch.tensor([0.2, 0.8, 0.5])
    tb_old = TensorBernoulli(
        _probs=probs_old,
        reinterpreted_batch_ndims=0,
        shape=probs_old.shape,
        device=probs_old.device,
    )
    td["dist"] = tb_old

    probs_new = torch.tensor([0.6, 0.4, 0.9])
    tb_new = TensorBernoulli(
        _probs=probs_new,
        reinterpreted_batch_ndims=0,
        shape=probs_new.shape,
        device=probs_new.device,
    )
    td["dist"] = tb_new

    # After reassignment, retrieval yields the new object
    assert td["dist"] is tb_new
    assert td["dist"] is not tb_old
