import pytest
import torch
from torch import Tensor
from torch.distributions import Distribution, Normal
from torch.distributions.transforms import ExpTransform

from tensorcontainer.tensor_distribution.base import TensorDistribution
from tensorcontainer.tensor_distribution.transformed_distribution import (
    TransformedDistribution,
)


class SimpleNormal(TensorDistribution):
    loc: Tensor
    scale: Tensor

    def dist(self) -> Distribution:
        return Normal(loc=self.loc, scale=self.scale)


class TestTransformedDistribution:
    """
    Tests the TransformedDistribution.

    This suite verifies that:
    - The distribution can be instantiated with valid parameters.
    - The distribution's sample shape is correct.
    """

    @pytest.mark.parametrize(
        "base_distribution, transforms",
        [
            (
                SimpleNormal(
                    loc=torch.tensor(0.0),
                    scale=torch.tensor(1.0),
                    shape=(),
                    device=torch.device("cpu"),
                ),
                [ExpTransform()],
            ),
        ],
    )
    def test_init(self, base_distribution, transforms):
        """
        Tests that the distribution can be instantiated with valid parameters.
        """
        dist = TransformedDistribution(
            base_distribution=base_distribution,
            transforms=transforms,
            shape=base_distribution.shape,
            device=base_distribution.device,
        )
        assert dist.base_distribution is base_distribution
        assert dist.transforms == transforms

    @pytest.mark.parametrize(
        "base_distribution, transforms, sample_shape",
        [
            (
                SimpleNormal(
                    loc=torch.tensor(0.0),
                    scale=torch.tensor(1.0),
                    shape=(),
                    device=torch.device("cpu"),
                ),
                [ExpTransform()],
                (1,),
            ),
        ],
    )
    def test_sample_shape(self, base_distribution, transforms, sample_shape):
        """
        Tests that the distribution's sample shape is correct.
        """
        dist = TransformedDistribution(
            base_distribution=base_distribution,
            transforms=transforms,
            shape=base_distribution.shape,
            device=base_distribution.device,
        )
        sample = dist.sample(sample_shape)
        expected_shape = sample_shape + base_distribution.shape
        assert sample.shape == expected_shape
