import pytest
import torch

from tensorcontainer.tensor_distribution.relaxed_categorical import RelaxedCategorical


class TestRelaxedCategorical:
    """
    Tests the RelaxedCategorical distribution.

    This suite verifies that:
    - The distribution can be instantiated with valid parameters.
    - The distribution's sample shape is correct.
    """

    @pytest.mark.parametrize(
        "temperature, probs",
        [
            (torch.tensor(0.1), torch.tensor([0.1, 0.9])),
            (torch.tensor([0.1, 0.2]), torch.tensor([[0.1, 0.9], [0.9, 0.1]])),
        ],
    )
    def test_init_probs(self, temperature, probs):
        """
        Tests that the distribution can be instantiated with probs.
        """
        dist = RelaxedCategorical(
            temperature=temperature,
            probs=probs,
            shape=probs.shape[:-1],
            device=probs.device,
        )
        assert torch.allclose(dist.temperature, temperature)
        assert dist.probs is not None
        assert torch.allclose(dist.probs, probs)

    @pytest.mark.parametrize(
        "temperature, logits",
        [
            (torch.tensor(0.1), torch.tensor([0.1, 0.9])),
            (torch.tensor([0.1, 0.2]), torch.tensor([[0.1, 0.9], [0.9, 0.1]])),
        ],
    )
    def test_init_logits(self, temperature, logits):
        """
        Tests that the distribution can be instantiated with logits.
        """
        dist = RelaxedCategorical(
            temperature=temperature,
            logits=logits,
            shape=logits.shape[:-1],
            device=logits.device,
        )
        assert torch.allclose(dist.temperature, temperature)
        assert dist.logits is not None
        assert torch.allclose(dist.logits, logits)

    @pytest.mark.parametrize(
        "temperature, probs, sample_shape",
        [
            (torch.tensor(0.1), torch.tensor([0.1, 0.9]), (1,)),
            (
                torch.tensor([0.1, 0.2]),
                torch.tensor([[0.1, 0.9], [0.9, 0.1]]),
                (5, 2),
            ),
        ],
    )
    def test_sample_shape(self, temperature, probs, sample_shape):
        """
        Tests that the distribution's sample shape is correct.
        """
        dist = RelaxedCategorical(
            temperature=temperature,
            probs=probs,
            shape=probs.shape[:-1],
            device=probs.device,
        )
        sample = dist.sample(sample_shape)
        expected_shape = sample_shape + probs.shape
        assert sample.shape == expected_shape
