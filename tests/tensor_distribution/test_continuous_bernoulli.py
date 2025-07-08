"""
Tests for the ContinuousBernoulli distribution.
"""

import pytest
import torch

from tensorcontainer.tensor_distribution.continuous_bernoulli import (
    ContinuousBernoulli,
)


class TestContinuousBernoulli:
    """
    Tests the ContinuousBernoulli distribution.

    This suite verifies that:
    - The distribution is initialized correctly.
    - Samples are drawn with the correct shape and values.
    - Log probabilities are calculated correctly.
    - The `view` method works as expected.
    """

    @pytest.mark.parametrize(
        "args,kwargs",
        [
            (
                {"_probs": torch.tensor(0.3), "_logits": torch.tensor(0.1)},
                {},
            ),  # both provided
        ],
    )
    def test_init_invalid_params(self, args, kwargs):
        """
        Tests that a ValueError is raised when both `probs` and `logits` are provided.
        """
        with pytest.raises(ValueError):
            ContinuousBernoulli(shape=(), device=torch.device("cpu"), **args, **kwargs)

    def test_sample_shape_and_dtype_and_values(self):
        """
        Tests that samples have the correct shape, dtype, and values.
        """
        probs = torch.rand(4, 3)
        dist = ContinuousBernoulli(
            _probs=probs,
            reinterpreted_batch_ndims=0,
            shape=probs.shape,
            device=probs.device,
        )
        # draw 5 i.i.d. samples
        samples = dist.sample(sample_shape=torch.Size([5]))
        # shape = (5, *batch_shape)
        assert samples.shape == (5, *probs.shape)
        assert samples.dtype == torch.float32
        # values must be between 0 and 1
        assert ((samples >= 0) & (samples <= 1)).all()

    @pytest.mark.parametrize(
        "rbn_dims,expected_shape",
        [
            (0, (2, 3)),  # no reinterpret → log_prob per-element
            (1, (2,)),  # sum over last 1 dim
            (2, ()),  # sum over last 2 dims → scalar
        ],
    )
    def test_log_prob_reinterpreted_batch_ndims(self, rbn_dims, expected_shape):
        """
        Tests that log_prob is calculated correctly with different `reinterpreted_batch_ndims`.
        """
        probs = torch.tensor([[0.2, 0.8, 0.1], [0.5, 0.5, 0.5]])
        dist = ContinuousBernoulli(
            _probs=probs,
            reinterpreted_batch_ndims=rbn_dims,
            shape=probs.shape,
            device=probs.device,
        )
        x = torch.tensor([[0.1, 0.9, 0.2], [0.6, 0.4, 0.8]])
        lp = dist.log_prob(x)
        # expected via torch.distributions
        td = torch.distributions.ContinuousBernoulli(probs)
        ref = td.log_prob(x)
        if rbn_dims > 0:
            ref = ref.sum(dim=list(range(len(ref.shape)))[-rbn_dims:])
        assert lp.shape == expected_shape
        assert torch.allclose(lp, ref)

    def test_init_logits(self):
        """
        Tests that the distribution can be initialized with `logits`.
        """
        logits = torch.randn(4, 3)
        dist = ContinuousBernoulli(
            _logits=logits, shape=logits.shape, device=logits.device
        )
        assert torch.allclose(dist.logits, logits)
        assert torch.allclose(dist.probs, torch.sigmoid(logits))

    def test_sample_logits(self):
        """
        Tests that samples can be drawn when the distribution is initialized with `logits`.
        """
        logits = torch.randn(4, 3)
        dist = ContinuousBernoulli(
            _logits=logits,
            reinterpreted_batch_ndims=0,
            shape=logits.shape,
            device=logits.device,
        )
        samples = dist.sample(sample_shape=torch.Size([5]))
        assert samples.shape == (5, *logits.shape)
        assert samples.dtype == torch.float32
        assert ((samples >= 0) & (samples <= 1)).all()

    def test_log_prob_logits(self):
        """
        Tests that log_prob is calculated correctly when the distribution is initialized with `logits`.
        """
        logits = torch.randn(2, 3)
        dist = ContinuousBernoulli(
            _logits=logits,
            reinterpreted_batch_ndims=0,
            shape=logits.shape,
            device=logits.device,
        )
        x = torch.tensor([[0.1, 0.9, 0.2], [0.6, 0.4, 0.8]])
        lp = dist.log_prob(x)
        td = torch.distributions.ContinuousBernoulli(logits=logits)
        ref = td.log_prob(x)
        assert torch.allclose(lp, ref)

    @pytest.mark.parametrize("shape", [(4,), (2, 2)])
    def test_view_probs(self, shape):
        """
        Tests that the `view` method works correctly when initialized with `probs`.
        """
        probs = torch.rand(*shape)
        dist = ContinuousBernoulli(_probs=probs, shape=probs.shape, device=probs.device)
        dist_view = dist.view(-1)
        assert dist_view.probs.shape == (probs.numel(),)

    @pytest.mark.parametrize("shape", [(4,), (2, 2)])
    def test_view_logits(self, shape):
        """
        Tests that the `view` method works correctly when initialized with `logits`.
        """
        logits = torch.randn(*shape)
        dist = ContinuousBernoulli(
            _logits=logits, shape=logits.shape, device=logits.device
        )
        dist_view = dist.view(-1)
        assert dist_view.logits.shape == (logits.numel(),)
