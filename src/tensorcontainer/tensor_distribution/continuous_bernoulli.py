from __future__ import annotations

from dataclasses import field
from typing import Optional

import torch
from torch import Tensor
from torch.distributions import ContinuousBernoulli as TorchContinuousBernoulli
from torch.distributions import Independent

from .base import TensorDistribution


class ContinuousBernoulli(TensorDistribution):
    """
    A Continuous Bernoulli distribution.

    This distribution is a continuous relaxation of the Bernoulli distribution, defined on the interval [0, 1].
    It is parameterized by either `probs` or `logits`.

    Source: https://pytorch.org/docs/stable/distributions.html#continuousbernoulli
    """

    _probs: Optional[Tensor] = field(default=None)
    _logits: Optional[Tensor] = field(default=None)
    reinterpreted_batch_ndims: int = 0

    @property
    def probs(self) -> Tensor:
        """
        Returns the probability of the Bernoulli distribution.
        """
        if self._probs is None:
            assert self._logits is not None
            self._probs = torch.sigmoid(self._logits)
        return self._probs

    @probs.setter
    def probs(self, value: Tensor):
        self._probs = value
        if value is not None:
            self._logits = None

    @property
    def logits(self) -> Tensor:
        """
        Returns the logits of the Bernoulli distribution.
        """
        if self._logits is None:
            assert self._probs is not None
            self._logits = torch.log(self._probs / (1 - self._probs + 1e-8))
        return self._logits

    @logits.setter
    def logits(self, value: Tensor):
        self._logits = value
        if value is not None:
            self._probs = None

    def dist(self) -> Independent:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        if self._probs is not None:
            return Independent(
                TorchContinuousBernoulli(
                    probs=self._probs,
                ),
                self.reinterpreted_batch_ndims,
            )
        else:
            return Independent(
                TorchContinuousBernoulli(
                    logits=self._logits,
                ),
                self.reinterpreted_batch_ndims,
            )
