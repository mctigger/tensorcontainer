from __future__ import annotations

from torch.distributions import Categorical
from torch.distributions import MixtureSameFamily as TorchMixtureSameFamily
from torch.distributions.distribution import Distribution

from tensorcontainer.tensor_distribution.base import TensorDistribution


class MixtureSameFamily(TensorDistribution):
    """
    Creates a mixture of distributions of the same family.

    Args:
        mixture_distribution (Categorical): The mixture distribution.
        component_distribution (TensorDistribution): The component distribution.
    """

    mixture_distribution: Categorical
    component_distribution: TensorDistribution

    def dist(self) -> Distribution:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return TorchMixtureSameFamily(
            mixture_distribution=self.mixture_distribution,
            component_distribution=self.component_distribution.dist(),
        )
