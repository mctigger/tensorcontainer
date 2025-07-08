"""
Tests for the HalfCauchy distribution.
"""

import pytest
import torch

from tensorcontainer.tensor_distribution.half_cauchy import HalfCauchy


class TestHalfCauchy:
    """
    Tests the HalfCauchy distribution.

    This suite verifies that:
    - The distribution is initialized correctly.
    - Samples are drawn with the correct shape.
    - Log probabilities are calculated correctly.
    - The `view` method works as expected.
    """

    def test_sample_shape_and_dtype(self):
        """
        Tests that samples have the correct shape and dtype.
        """
        scale = torch.rand(4, 3)
        dist = HalfCauchy(
            scale=scale,
            reinterpreted_batch_ndims=0,
            shape=scale.shape,
            device=scale.device,
        )
        # draw 5 i.i.d. samples
        samples = dist.sample(sample_shape=torch.Size([5]))
        # shape = (5, *batch_shape)
        assert samples.shape == (5, *scale.shape)
        assert samples.dtype == torch.float32

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
        scale = torch.rand(2, 3)
        dist = HalfCauchy(
            scale=scale,
            reinterpreted_batch_ndims=rbn_dims,
            shape=scale.shape,
            device=scale.device,
        )
        x = dist.sample()
        lp = dist.log_prob(x)
        # expected via torch.distributions
        td = torch.distributions.HalfCauchy(scale)
        ref = td.log_prob(x)
        if rbn_dims > 0:
            ref = ref.sum(dim=list(range(len(ref.shape)))[-rbn_dims:])
        assert lp.shape == expected_shape
        assert torch.allclose(lp, ref)

    @pytest.mark.parametrize("shape", [(4,), (2, 2)])
    def test_view(self, shape):
        """
        Tests that the `view` method works correctly.
        """
        scale = torch.rand(*shape)
        dist = HalfCauchy(scale=scale, shape=scale.shape, device=scale.device)
        dist_view = dist.view(-1)
        assert dist_view.scale.shape == (scale.numel(),)
