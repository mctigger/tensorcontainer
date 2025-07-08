from __future__ import annotations

from typing import Optional

from torch import Tensor
from torch.distributions import RelaxedBernoulli as TorchRelaxedBernoulli
from torch.distributions.distribution import Distribution

from tensorcontainer.tensor_distribution.base import TensorDistribution


class RelaxedBernoulli(TensorDistribution):
    """
    Creates a RelaxedBernoulli distribution parameterized by `temperature`, `probs` or `logits`.

    Args:
        temperature (Tensor): The temperature of the distribution.
        probs (Optional[Tensor]): The probabilities of the categories.
        logits (Optional[Tensor]): The logits of the categories.
    """

    temperature: Tensor
    probs: Optional[Tensor] = None
    logits: Optional[Tensor] = None

    def dist(self) -> Distribution:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return TorchRelaxedBernoulli(
            temperature=self.temperature,
            probs=self.probs,
            logits=self.logits,
        )
