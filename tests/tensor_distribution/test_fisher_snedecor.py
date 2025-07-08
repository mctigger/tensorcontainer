"""
Tests for the FisherSnedecor distribution.
"""

import pytest
import torch

from tensorcontainer.tensor_distribution.fisher_snedecor import FisherSnedecor


class TestFisherSnedecor:
    """
    Tests the FisherSnedecor distribution.

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
        df1 = torch.rand(4, 3) * 5 + 1
        df2 = torch.rand(4, 3) * 5 + 1
        dist = FisherSnedecor(
            df1=df1,
            df2=df2,
            reinterpreted_batch_ndims=0,
            shape=df1.shape,
            device=df1.device,
        )
        # draw 5 i.i.d. samples
        samples = dist.sample(sample_shape=torch.Size([5]))
        # shape = (5, *batch_shape)
        assert samples.shape == (5, *df1.shape)
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
        df1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        df2 = torch.tensor([[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])
        dist = FisherSnedecor(
            df1=df1,
            df2=df2,
            reinterpreted_batch_ndims=rbn_dims,
            shape=df1.shape,
            device=df1.device,
        )
        x = dist.sample()
        lp = dist.log_prob(x)
        # expected via torch.distributions
        td = torch.distributions.FisherSnedecor(df1, df2)
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
        df1 = torch.rand(*shape) * 5 + 1
        df2 = torch.rand(*shape) * 5 + 1
        dist = FisherSnedecor(df1=df1, df2=df2, shape=df1.shape, device=df1.device)
        dist_view = dist.view(-1)
        assert dist_view.df1.shape == (df1.numel(),)
        assert dist_view.df2.shape == (df2.numel(),)
