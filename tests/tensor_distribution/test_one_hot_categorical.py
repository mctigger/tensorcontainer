import pytest
import torch

from tensorcontainer.tensor_distribution.one_hot_categorical import OneHotCategorical


class TestOneHotCategorical:
    """
    Tests the OneHotCategorical distribution.

    This suite verifies that:
    - The distribution can be instantiated with valid parameters.
    - The distribution raises errors for invalid parameters.
    - The distribution's sample shape is correct.
    """

    @pytest.mark.parametrize(
        "probs",
        [
            torch.tensor([0.1, 0.9]),
            torch.tensor([[0.1, 0.9], [0.9, 0.1]]),
        ],
    )
    def test_init_probs(self, probs):
        """
        Tests that the distribution can be instantiated with probs.
        """
        dist = OneHotCategorical(
            probs=probs, shape=probs.shape[:-1], device=probs.device
        )
        assert dist.probs is not None
        assert torch.allclose(dist.probs, probs)

    @pytest.mark.parametrize(
        "logits",
        [
            torch.tensor([0.1, 0.9]),
            torch.tensor([[0.1, 0.9], [0.9, 0.1]]),
        ],
    )
    def test_init_logits(self, logits):
        """
        Tests that the distribution can be instantiated with logits.
        """
        dist = OneHotCategorical(
            logits=logits, shape=logits.shape[:-1], device=logits.device
        )
        assert dist.logits is not None
        assert torch.allclose(dist.logits, logits)

    @pytest.mark.parametrize(
        "probs, sample_shape",
        [
            (torch.tensor([0.1, 0.9]), (1,)),
            (torch.tensor([[0.1, 0.9], [0.9, 0.1]]), (5, 2)),
        ],
    )
    def test_sample_shape(self, probs, sample_shape):
        """
        Tests that the distribution's sample shape is correct.
        """
        dist = OneHotCategorical(
            probs=probs, shape=probs.shape[:-1], device=probs.device
        )
        sample = dist.sample(sample_shape)
        expected_shape = sample_shape + probs.shape
        assert sample.shape == expected_shape
