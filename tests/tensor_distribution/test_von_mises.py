"""
Tests for the VonMises distribution.
"""

import pytest
import torch

from tensorcontainer.tensor_distribution.von_mises import VonMises


class TestVonMises:
    """
    Tests the VonMises distribution.

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
        loc = torch.randn(4, 3)
        concentration = torch.rand(4, 3)
        dist = VonMises(
            loc=loc,
            concentration=concentration,
            reinterpreted_batch_ndims=0,
            shape=loc.shape,
            device=loc.device,
        )
        # draw 5 i.i.d. samples
        samples = dist.sample(sample_shape=torch.Size([5]))
        # shape = (5, *batch_shape)
        assert samples.shape == (5, *loc.shape)
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
        loc = torch.randn(2, 3)
        concentration = torch.rand(2, 3)
        dist = VonMises(
            loc=loc,
            concentration=concentration,
            reinterpreted_batch_ndims=rbn_dims,
            shape=loc.shape,
            device=loc.device,
        )
        x = dist.sample()
        lp = dist.log_prob(x)
        # expected via torch.distributions
        td = torch.distributions.VonMises(loc, concentration)
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
        loc = torch.randn(*shape)
        concentration = torch.rand(*shape)
        dist = VonMises(
            loc=loc,
            concentration=concentration,
            shape=loc.shape,
            device=loc.device,
        )
        dist_view = dist.view(-1)
        assert dist_view.loc.shape == (loc.numel(),)
        assert dist_view.concentration.shape == (concentration.numel(),)
