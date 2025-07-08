import pytest
import torch

from tensorcontainer.tensor_distribution.logistic_normal import LogisticNormal


class TestLogisticNormal:
    """
    Tests the LogisticNormal distribution.

    This suite verifies that:
    - The distribution can be instantiated with valid parameters.
    - The distribution raises errors for invalid parameters.
    - The distribution's sample shape is correct.
    """

    @pytest.mark.parametrize(
        "loc, scale",
        [
            (torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])),
            (
                torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
                torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
            ),
        ],
    )
    def test_init(self, loc, scale):
        """
        Tests that the distribution can be instantiated with valid parameters.
        """
        dist = LogisticNormal(
            loc=loc, scale=scale, shape=loc.shape[:-1], device=loc.device
        )
        assert torch.allclose(dist.loc, loc)
        assert torch.allclose(dist.scale, scale)

    @pytest.mark.parametrize(
        "loc, scale",
        [
            (torch.tensor([0.0, 0.0]), torch.tensor([-1.0, 1.0])),
        ],
    )
    def test_init_invalid_scale(self, loc, scale):
        """
        Tests that the distribution raises a ValueError for invalid scale.
        """
        with pytest.raises(ValueError):
            LogisticNormal(
                loc=loc, scale=scale, shape=loc.shape[:-1], device=loc.device
            )

    @pytest.mark.parametrize(
        "loc, scale, sample_shape",
        [
            (torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]), (1,)),
            (
                torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
                torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
                (5, 2),
            ),
        ],
    )
    def test_sample_shape(self, loc, scale, sample_shape):
        """
        Tests that the distribution's sample shape is correct.
        """
        dist = LogisticNormal(
            loc=loc, scale=scale, shape=loc.shape[:-1], device=loc.device
        )
        sample = dist.sample(sample_shape)
        expected_shape = sample_shape + loc.shape[:-1] + (loc.shape[-1] + 1,)
        assert sample.shape == expected_shape
