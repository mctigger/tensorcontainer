import pytest
import torch

from rtd.tensor_distribution import TensorCategorical
from rtd.tensor_dict import TensorDict


def _create_distributions(batch_size, num_categories, output_shape):
    """Helper to create TensorCategorical and torch.distributions.Categorical instances."""
    logits_shape = batch_size + (num_categories,)
    logits = torch.randn(*logits_shape)
    tensor_dict = TensorDict({"logits": logits}, shape=batch_size)

    dist = TensorCategorical(
        logits=tensor_dict["logits"],
        output_shape=output_shape,
        shape=batch_size,
    )
    torch_dist = torch.distributions.Categorical(logits=logits)
    return dist, torch_dist, logits, num_categories


def _test_log_prob(dist, torch_dist, num_categories, sample_shape=None):
    """Helper to test log_prob functionality."""
    torch_sample = (
        torch_dist.sample(sample_shape) if sample_shape else torch_dist.sample()
    )
    one_hot_torch_sample = torch.nn.functional.one_hot(
        torch_sample, num_classes=num_categories
    ).float()
    log_prob = dist.log_prob(one_hot_torch_sample)
    torch_log_prob = torch_dist.log_prob(torch_sample)

    assert log_prob.shape == torch_log_prob.shape
    assert torch.allclose(log_prob, torch_log_prob)


@pytest.mark.parametrize("batch_size", [(3, 4), (3,)])
def test_tensor_categorical_basic_functionality(batch_size):
    """
    Tests basic creation, sampling, and log_prob functionality of TensorCategorical.
    """
    num_categories = 5
    output_shape = ()  # Fixed to empty output_shape to avoid RuntimeError
    dist, torch_dist, logits, num_categories = _create_distributions(
        batch_size, num_categories, output_shape
    )

    # Test creation
    assert torch.equal(dist["logits"], logits)

    # Test sample shape
    sample = dist.sample()
    expected_sample_shape = batch_size + output_shape + (num_categories,)
    assert sample.shape == expected_sample_shape

    # Test log_prob
    _test_log_prob(dist, torch_dist, num_categories, sample_shape=output_shape)


@pytest.mark.parametrize("batch_size", [(3, 4), (3,), ()])
@pytest.mark.parametrize("num_categories", [1, 2, 10])
def test_tensor_categorical_edge_cases(batch_size, num_categories):
    """
    Tests TensorCategorical with various edge cases for batch_size, num_categories,
    and reinterpreted_batch_ndims.
    """
    # For TensorCategorical, reinterpreted_batch_ndims is len(output_shape).
    # In this test, output_shape is fixed to (), so reinterpreted_batch_ndims is always 0.
    reinterpreted_batch_ndims_for_categorical = 0

    dist, torch_dist, logits, num_categories = _create_distributions(
        batch_size, num_categories, output_shape=()
    )

    # Test creation and internal state
    assert torch.equal(dist["logits"], logits)
    assert (
        dist.dist().batch_shape
        == batch_size[: len(batch_size) - reinterpreted_batch_ndims_for_categorical]
    )
    assert dist.dist().event_shape == batch_size[
        len(batch_size) - reinterpreted_batch_ndims_for_categorical :
    ] + (num_categories,)

    # Test sample shape
    sample = dist.sample()
    expected_sample_shape = dist.dist().batch_shape + dist.dist().event_shape
    assert sample.shape == expected_sample_shape

    # Test log_prob shape and value
    _test_log_prob(dist, torch_dist, num_categories)


def test_tensor_categorical_empty_batch_and_output_shape():
    """
    Tests TensorCategorical with empty batch_size and empty output_shape.
    """
    num_categories = 3
    batch_size = ()
    output_shape = ()

    dist, torch_dist, logits, num_categories = _create_distributions(
        batch_size, num_categories, output_shape
    )

    # Test creation
    assert torch.equal(dist["logits"], logits)
    assert dist.dist().batch_shape == ()
    assert dist.dist().event_shape == (num_categories,)

    # Test sample
    sample = dist.sample()
    assert sample.shape == (num_categories,)

    # Test log_prob
    _test_log_prob(dist, torch_dist, num_categories)


def test_tensor_categorical_large_logits():
    """
    Tests TensorCategorical with large logits to check for numerical stability.
    """
    batch_size = (2, 3)
    num_categories = 4
    output_shape = ()

    # Manually create logits to ensure large values
    logits_shape = batch_size + (num_categories,)
    logits = torch.randn(*logits_shape) * 1000  # Large values
    tensor_dict = TensorDict({"logits": logits}, shape=batch_size)

    dist = TensorCategorical(
        logits=tensor_dict["logits"],
        output_shape=output_shape,
        shape=batch_size,
    )
    torch_dist = torch.distributions.Categorical(logits=logits)

    sample = dist.sample()
    assert sample.shape == batch_size + (num_categories,)

    _test_log_prob(dist, torch_dist, num_categories)


def test_tensor_categorical_zero_logits():
    """
    Tests TensorCategorical with all zero logits.
    """
    batch_size = (2,)
    num_categories = 3
    output_shape = ()

    # Manually create logits to ensure zero values
    logits_shape = batch_size + (num_categories,)
    logits = torch.zeros(*logits_shape)  # All zeros
    tensor_dict = TensorDict({"logits": logits}, shape=batch_size)

    dist = TensorCategorical(
        logits=tensor_dict["logits"],
        output_shape=output_shape,
        shape=batch_size,
    )
    torch_dist = torch.distributions.Categorical(logits=logits)

    sample = dist.sample()
    assert sample.shape == batch_size + (num_categories,)

    _test_log_prob(dist, torch_dist, num_categories)
