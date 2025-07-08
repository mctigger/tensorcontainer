from __future__ import annotations

from typing import Optional

from torch import Tensor
from torch.distributions import RelaxedOneHotCategorical as TorchRelaxedCategorical
from torch.distributions.distribution import Distribution

from tensorcontainer.tensor_distribution.base import TensorDistribution


class RelaxedCategorical(TensorDistribution):
    """
    Creates a RelaxedCategorical distribution parameterized by `temperature`, `probs` or `logits`.

    This is a relaxed version of the `OneHotCategorical` distribution, so the name of the
    torch distribution is `RelaxedOneHotCategorical`.

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
        return TorchRelaxedCategorical(
            temperature=self.temperature,
            probs=self.probs,
            logits=self.logits,
        )
