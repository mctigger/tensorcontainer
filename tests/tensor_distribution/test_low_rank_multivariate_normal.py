import pytest
import torch

from tensorcontainer.tensor_distribution.low_rank_multivariate_normal import (
    LowRankMultivariateNormal,
)


class TestLowRankMultivariateNormal:
    """
    Tests the LowRankMultivariateNormal distribution.

    This suite verifies that:
    - The distribution can be instantiated with valid parameters.
    - The distribution raises errors for invalid parameters.
    - The distribution's sample shape is correct.
    """

    @pytest.mark.parametrize(
        "loc, cov_factor, cov_diag",
        [
            (
                torch.randn(2),
                torch.randn(2, 1),
                torch.rand(2),
            ),
            (
                torch.randn(3, 2),
                torch.randn(3, 2, 1),
                torch.rand(3, 2),
            ),
        ],
    )
    def test_init(self, loc, cov_factor, cov_diag):
        """
        Tests that the distribution can be instantiated with valid parameters.
        """
        dist = LowRankMultivariateNormal(
            loc=loc,
            cov_factor=cov_factor,
            cov_diag=cov_diag,
            shape=loc.shape[:-1],
            device=loc.device,
        )
        assert torch.allclose(dist.loc, loc)
        assert torch.allclose(dist.cov_factor, cov_factor)
        assert torch.allclose(dist.cov_diag, cov_diag)

    @pytest.mark.parametrize(
        "loc, cov_factor, cov_diag",
        [
            (
                torch.randn(2),
                torch.randn(2, 1),
                torch.zeros(2),
            ),
            (
                torch.randn(3, 2),
                torch.randn(3, 2, 1),
                -torch.rand(3, 2),
            ),
        ],
    )
    def test_init_invalid_cov_diag(self, loc, cov_factor, cov_diag):
        """
        Tests that the distribution raises a ValueError for invalid cov_diag.
        """
        with pytest.raises(ValueError):
            LowRankMultivariateNormal(
                loc=loc,
                cov_factor=cov_factor,
                cov_diag=cov_diag,
                shape=loc.shape[:-1],
                device=loc.device,
            )

    @pytest.mark.parametrize(
        "loc, cov_factor, cov_diag, sample_shape",
        [
            (
                torch.randn(2),
                torch.randn(2, 1),
                torch.rand(2),
                (1,),
            ),
            (
                torch.randn(3, 2),
                torch.randn(3, 2, 1),
                torch.rand(3, 2),
                (5, 2),
            ),
        ],
    )
    def test_sample_shape(self, loc, cov_factor, cov_diag, sample_shape):
        """
        Tests that the distribution's sample shape is correct.
        """
        dist = LowRankMultivariateNormal(
            loc=loc,
            cov_factor=cov_factor,
            cov_diag=cov_diag,
            shape=loc.shape[:-1],
            device=loc.device,
        )
        sample = dist.sample(sample_shape)
        expected_shape = sample_shape + loc.shape
        assert sample.shape == expected_shape
