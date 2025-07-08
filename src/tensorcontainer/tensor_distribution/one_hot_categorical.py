from __future__ import annotations

from typing import Optional

from torch import Tensor
from torch.distributions import OneHotCategorical as TorchOneHotCategorical
from torch.distributions.distribution import Distribution

from tensorcontainer.tensor_distribution.base import TensorDistribution


class OneHotCategorical(TensorDistribution):
    """
    Creates a one-hot categorical distribution parameterized by `probs` or `logits`.

    Args:
        probs (Optional[Tensor]): The probabilities of the categories.
        logits (Optional[Tensor]): The logits of the categories.
    """

    probs: Optional[Tensor] = None
    logits: Optional[Tensor] = None

    def dist(self) -> Distribution:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return TorchOneHotCategorical(
            probs=self.probs,
            logits=self.logits,
        )
