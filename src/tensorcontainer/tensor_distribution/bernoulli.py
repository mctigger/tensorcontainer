from __future__ import annotations

from dataclasses import field
from typing import Optional

import torch
from torch import Tensor
from torch.distributions import Independent

from .base import TensorDistribution


class TensorBernoulli(TensorDistribution):
    _probs: Optional[Tensor] = field(default=None)
    _logits: Optional[Tensor] = field(default=None)
    reinterpreted_batch_ndims: int = 0

    def __post_init__(self):
        super().__post_init__()  # Call parent's post_init
        if (self._probs is None) == (self._logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )

    @property
    def probs(self):
        if self._probs is None:
            assert self._logits is not None
            self._probs = torch.sigmoid(self._logits)
        return self._probs

    @probs.setter
    def probs(self, value):
        self._probs = value
        if value is not None:
            self._logits = None

    @property
    def logits(self):
        if self._logits is None:
            assert self._probs is not None
            self._logits = torch.log(self._probs / (1 - self._probs + 1e-8))
        return self._logits

    @logits.setter
    def logits(self, value):
        self._logits = value
        if value is not None:
            self._probs = None

    def dist(self):
        if self._probs is not None:
            return Independent(
                torch.distributions.Bernoulli(
                    probs=self._probs,
                ),
                self.reinterpreted_batch_ndims,
            )
        else:
            return Independent(
                torch.distributions.Bernoulli(
                    logits=self._logits,
                ),
                self.reinterpreted_batch_ndims,
            )
