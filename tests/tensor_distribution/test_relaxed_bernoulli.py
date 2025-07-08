import pytest
import torch

from tensorcontainer.tensor_distribution.relaxed_bernoulli import RelaxedBernoulli


class TestRelaxedBernoulli:
    """
    Tests the RelaxedBernoulli distribution.

    This suite verifies that:
    - The distribution can be instantiated with valid parameters.
    - The distribution's sample shape is correct.
    """

    @pytest.mark.parametrize(
        "temperature, probs",
        [
            (torch.tensor(0.1), torch.tensor(0.5)),
            (torch.tensor([0.1, 0.2]), torch.tensor([0.5, 0.5])),
        ],
    )
    def test_init_probs(self, temperature, probs):
        """
        Tests that the distribution can be instantiated with probs.
        """
        dist = RelaxedBernoulli(
            temperature=temperature,
            probs=probs,
            shape=probs.shape,
            device=probs.device,
        )
        assert torch.allclose(dist.temperature, temperature)
        assert dist.probs is not None
        assert torch.allclose(dist.probs, probs)

    @pytest.mark.parametrize(
        "temperature, logits",
        [
            (torch.tensor(0.1), torch.tensor(0.0)),
            (torch.tensor([0.1, 0.2]), torch.tensor([0.0, 0.0])),
        ],
    )
    def test_init_logits(self, temperature, logits):
        """
        Tests that the distribution can be instantiated with logits.
        """
        dist = RelaxedBernoulli(
            temperature=temperature,
            logits=logits,
            shape=logits.shape,
            device=logits.device,
        )
        assert torch.allclose(dist.temperature, temperature)
        assert dist.logits is not None
        assert torch.allclose(dist.logits, logits)

    @pytest.mark.parametrize(
        "temperature, probs, sample_shape",
        [
            (torch.tensor(0.1), torch.tensor(0.5), (1,)),
            (torch.tensor([0.1, 0.2]), torch.tensor([0.5, 0.5]), (5, 2)),
        ],
    )
    def test_sample_shape(self, temperature, probs, sample_shape):
        """
        Tests that the distribution's sample shape is correct.
        """
        dist = RelaxedBernoulli(
            temperature=temperature,
            probs=probs,
            shape=probs.shape,
            device=probs.device,
        )
        sample = dist.sample(sample_shape)
        expected_shape = sample_shape + probs.shape
        assert sample.shape == expected_shape
