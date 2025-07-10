from __future__ import annotations

from torch import Tensor
from torch.distributions import LogisticNormal as TorchLogisticNormal
from torch.distributions.distribution import Distribution

from tensorcontainer.tensor_distribution.base import TensorDistribution


class LogisticNormal(TensorDistribution):
    """
    Creates a logistic-normal distribution.

    Args:
        loc (Tensor): The mean of the underlying Normal distribution.
        scale (Tensor): The standard deviation of the underlying Normal distribution.
    """

    loc: Tensor
    scale: Tensor

    def dist(self) -> Distribution:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return TorchLogisticNormal(
            loc=self.loc,
            scale=self.scale,
        )
