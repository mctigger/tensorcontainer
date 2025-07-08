import pytest
import torch
from torch import Tensor
from torch.distributions import Categorical, Distribution, Normal

from tensorcontainer.tensor_distribution.base import TensorDistribution
from tensorcontainer.tensor_distribution.mixture_same_family import (
    MixtureSameFamily,
)


class SimpleNormal(TensorDistribution):
    loc: Tensor
    scale: Tensor

    def dist(self) -> Distribution:
        return Normal(loc=self.loc, scale=self.scale)


class TestMixtureSameFamily:
    """
    Tests the MixtureSameFamily distribution.

    This suite verifies that:
    - The distribution can be instantiated with valid parameters.
    - The distribution's sample shape is correct.
    """

    @pytest.mark.parametrize(
        "mixture_distribution, component_distribution",
        [
            (
                Categorical(probs=torch.tensor([0.5, 0.5])),
                SimpleNormal(
                    loc=torch.tensor([0.0, 1.0]),
                    scale=torch.tensor([1.0, 1.0]),
                    shape=(2,),
                    device=torch.device("cpu"),
                ),
            ),
        ],
    )
    def test_init(self, mixture_distribution, component_distribution):
        """
        Tests that the distribution can be instantiated with valid parameters.
        """
        dist = MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
            shape=mixture_distribution.batch_shape,
            device=torch.device("cpu"),
        )
        assert dist.mixture_distribution is mixture_distribution
        assert dist.component_distribution is component_distribution

    @pytest.mark.parametrize(
        "mixture_distribution, component_distribution, sample_shape",
        [
            (
                Categorical(probs=torch.tensor([0.5, 0.5])),
                SimpleNormal(
                    loc=torch.tensor([0.0, 1.0]),
                    scale=torch.tensor([1.0, 1.0]),
                    shape=(2,),
                    device=torch.device("cpu"),
                ),
                (1,),
            ),
        ],
    )
    def test_sample_shape(
        self, mixture_distribution, component_distribution, sample_shape
    ):
        """
        Tests that the distribution's sample shape is correct.
        """
        dist = MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
            shape=mixture_distribution.batch_shape,
            device=torch.device("cpu"),
        )
        sample = dist.sample(sample_shape)
        expected_shape = sample_shape + component_distribution.dist().event_shape
        assert sample.shape == expected_shape
