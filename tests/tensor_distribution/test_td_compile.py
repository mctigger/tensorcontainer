import pytest
import torch

from src.rtd.tensor_distribution import (
    TensorBernoulli,
    TensorCategorical,
    TensorNormal,
)
from tests.tensor_dict.compile_utils import run_and_compare_compiled


@pytest.fixture(
    params=[
        ((), ()),
        ((2, 3), ()),
        ((), (4, 5)),
        ((2, 3), (4, 5)),
    ],
    ids=[
        "scalar_batch_scalar_event",
        "multi_batch_scalar_event",
        "scalar_batch_multi_event",
        "multi_batch_multi_event",
    ],
)
def shapes(request):
    """Provides parametrized batch and event shapes for tests."""
    return request.param


class TestTensorDistributions:
    """A test suite for TensorDistribution subclasses."""

    def test_normal(self, shapes):
        """Tests the TensorNormal distribution."""
        batch_shape, event_shape = shapes
        full_shape = batch_shape + event_shape
        loc = torch.randn(full_shape)
        scale = torch.exp(torch.randn(full_shape))
        reinterpreted_batch_ndims = len(event_shape)

        td_normal = TensorNormal(
            loc=loc,
            scale=scale,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
            shape=batch_shape,
        )

        # Test sampling methods
        run_and_compare_compiled(td_normal.rsample)
        run_and_compare_compiled(td_normal.sample)

        # Test other operations
        run_and_compare_compiled(td_normal.entropy)
        sample_val = td_normal.sample()
        run_and_compare_compiled(td_normal.log_prob, sample_val)

        # Check properties
        assert td_normal.mean.shape == full_shape
        assert td_normal.stddev.shape == full_shape
        assert td_normal.mode.shape == full_shape

    def test_bernoulli(self, shapes):
        """Tests the TensorBernoulli distribution."""
        batch_shape, event_shape = shapes
        full_shape = batch_shape + event_shape
        probs = torch.sigmoid(torch.randn(full_shape))
        reinterpreted_batch_ndims = len(event_shape)

        td_bernoulli = TensorBernoulli(
            probs=probs,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
            shape=batch_shape,
        )

        # Test sampling (Bernoulli does not have rsample, this is a key fix)
        run_and_compare_compiled(td_bernoulli.sample)

        # Test other operations
        run_and_compare_compiled(td_bernoulli.entropy)
        sample_val = td_bernoulli.sample()
        run_and_compare_compiled(td_bernoulli.log_prob, sample_val)

        # Check properties
        assert td_bernoulli.mean.shape == full_shape

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    @pytest.mark.parametrize("num_classes", [2, 10])
    def test_categorical(self, batch_shape, num_classes):
        """
        Tests the original TensorCategorical distribution.
        NOTE: This test is constrained by the original implementation, which
        only works correctly when the distribution's event_shape (passed as
        `output_shape`) is empty.
        """
        # The original implementation only supports an empty event shape.
        event_shape = ()

        logits_shape = batch_shape + event_shape + (num_classes,)
        logits = torch.randn(logits_shape)

        td_categorical = TensorCategorical(
            logits=logits,
            output_shape=event_shape,  # This must be Size(()) for the original code
            shape=batch_shape,
        )

        # Test sampling methods
        run_and_compare_compiled(td_categorical.rsample)
        run_and_compare_compiled(td_categorical.sample)

        # Test other operations
        run_and_compare_compiled(td_categorical.entropy)
        sample_val = td_categorical.sample()
        run_and_compare_compiled(td_categorical.log_prob, sample_val)

        # Check properties
        expected_sample_shape = batch_shape + event_shape + (num_classes,)
        assert td_categorical.sample().shape == expected_sample_shape
        assert td_categorical.mean.shape == expected_sample_shape
