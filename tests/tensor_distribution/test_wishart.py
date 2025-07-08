import pytest
import torch

from tensorcontainer.tensor_distribution.wishart import Wishart


class TestWishart:
    """
    Tests the Wishart distribution.

    This suite verifies that:
    - The distribution can be instantiated with valid parameters.
    - The distribution raises errors for invalid parameters.
    - The distribution's sample shape is correct.
    """

    @pytest.mark.parametrize(
        "df, covariance_matrix",
        [
            (
                torch.tensor(3.0),
                torch.eye(3),
            ),
            (
                torch.tensor([3.0, 4.0]),
                torch.stack([torch.eye(3), torch.eye(3)]),
            ),
        ],
    )
    def test_init_covariance_matrix(self, df, covariance_matrix):
        """
        Tests that the distribution can be instantiated with a covariance matrix.
        """
        dist = Wishart(
            df=df,
            covariance_matrix=covariance_matrix,
            shape=df.shape,
            device=df.device,
        )
        assert torch.allclose(dist.df, df)
        assert dist.covariance_matrix is not None
        assert torch.allclose(dist.covariance_matrix, covariance_matrix)

    @pytest.mark.parametrize(
        "df, precision_matrix",
        [
            (
                torch.tensor(3.0),
                torch.eye(3),
            ),
            (
                torch.tensor([3.0, 4.0]),
                torch.stack([torch.eye(3), torch.eye(3)]),
            ),
        ],
    )
    def test_init_precision_matrix(self, df, precision_matrix):
        """
        Tests that the distribution can be instantiated with a precision matrix.
        """
        dist = Wishart(
            df=df,
            precision_matrix=precision_matrix,
            shape=df.shape,
            device=df.device,
        )
        assert torch.allclose(dist.df, df)
        assert dist.precision_matrix is not None
        assert torch.allclose(dist.precision_matrix, precision_matrix)

    @pytest.mark.parametrize(
        "df, scale_tril",
        [
            (
                torch.tensor(3.0),
                torch.eye(3),
            ),
            (
                torch.tensor([3.0, 4.0]),
                torch.stack([torch.eye(3), torch.eye(3)]),
            ),
        ],
    )
    def test_init_scale_tril(self, df, scale_tril):
        """
        Tests that the distribution can be instantiated with a scale_tril.
        """
        dist = Wishart(
            df=df,
            scale_tril=scale_tril,
            shape=df.shape,
            device=df.device,
        )
        assert torch.allclose(dist.df, df)
        assert dist.scale_tril is not None
        assert torch.allclose(dist.scale_tril, scale_tril)

    @pytest.mark.parametrize(
        "df, covariance_matrix",
        [
            (
                torch.tensor(0.0),
                torch.eye(3),
            ),
            (
                torch.tensor(-1.0),
                torch.eye(3),
            ),
        ],
    )
    def test_init_invalid_df(self, df, covariance_matrix):
        """
        Tests that the distribution raises a ValueError for invalid df.
        """
        with pytest.raises(ValueError):
            Wishart(
                df=df,
                covariance_matrix=covariance_matrix,
                shape=df.shape,
                device=df.device,
            )

    @pytest.mark.parametrize(
        "df, covariance_matrix, sample_shape",
        [
            (
                torch.tensor(3.0),
                torch.eye(3),
                (1,),
            ),
            (
                torch.tensor([3.0, 4.0]),
                torch.stack([torch.eye(3), torch.eye(3)]),
                (5, 2),
            ),
        ],
    )
    def test_sample_shape(self, df, covariance_matrix, sample_shape):
        """
        Tests that the distribution's sample shape is correct.
        """
        dist = Wishart(
            df=df,
            covariance_matrix=covariance_matrix,
            shape=df.shape,
            device=df.device,
        )
        sample = dist.sample(sample_shape)
        expected_shape = sample_shape + df.shape + (3, 3)
        assert sample.shape == expected_shape
