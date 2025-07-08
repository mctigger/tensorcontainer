import pytest
import torch

from tensorcontainer.tensor_distribution.lkj_cholesky import LKJCholesky


class TestLKJCholesky:
    """
    Tests the LKJCholesky distribution.

    This suite verifies that:
    - The distribution can be instantiated with valid parameters.
    - The distribution raises errors for invalid parameters.
    - The distribution's sample shape is correct.
    """

    @pytest.mark.parametrize(
        "dim, concentration",
        [
            (2, torch.tensor(1.0)),
            (3, torch.tensor([1.0, 2.0])),
            (4, torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        ],
    )
    def test_init(self, dim, concentration):
        """
        Tests that the distribution can be instantiated with valid parameters.
        """
        dist = LKJCholesky(
            dimension=dim,
            concentration=concentration,
            shape=concentration.shape,
            device=concentration.device,
        )
        assert dist.dimension == dim
        assert torch.allclose(dist.concentration, concentration)

    @pytest.mark.parametrize(
        "dim, concentration",
        [
            (2, torch.tensor(0.0)),
            (3, torch.tensor(-1.0)),
        ],
    )
    def test_init_invalid_concentration(self, dim, concentration):
        """
        Tests that the distribution raises a ValueError for invalid concentration.
        """
        with pytest.raises(ValueError):
            LKJCholesky(
                dimension=dim,
                concentration=concentration,
                shape=concentration.shape,
                device=concentration.device,
            )

    @pytest.mark.parametrize(
        "dim, concentration, sample_shape",
        [
            (2, torch.tensor(1.0), (1,)),
            (3, torch.tensor([1.0, 2.0]), (5, 2)),
            (4, torch.tensor([[1.0, 2.0], [3.0, 4.0]]), ()),
        ],
    )
    def test_sample_shape(self, dim, concentration, sample_shape):
        """
        Tests that the distribution's sample shape is correct.
        """
        dist = LKJCholesky(
            dimension=dim,
            concentration=concentration,
            shape=concentration.shape,
            device=concentration.device,
        )
        sample = dist.sample(sample_shape)
        expected_shape = sample_shape + concentration.shape + (dim, dim)
        assert sample.shape == expected_shape
