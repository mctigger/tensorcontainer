from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.distributions import Independent

from tensorcontainer.distributions.soft_bernoulli import SoftBernoulli
from tensorcontainer.tensor_dict import TDCompatible

from .base import TensorDistribution


class TensorSoftBernoulli(TensorDistribution):
    def __init__(
        self,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            assert probs is not None
            shape = probs.shape
            device = probs.device
        else:
            assert logits is not None
            shape = logits.shape
            device = logits.device

        super().__init__(shape, device)
        self._probs = probs
        self._logits = logits

    @classmethod
    def _unflatten_distribution(
        cls, tensor_attributes: Dict[str, TDCompatible], meta_attributes: Dict[str, Any]
    ):
        return cls(
            probs=tensor_attributes.get("_probs"),
            logits=tensor_attributes.get("_logits"),
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
                SoftBernoulli(
                    probs=self._probs,
                ),
                0,
            )
        else:
            return Independent(
                SoftBernoulli(
                    logits=self._logits,
                ),
                0,
            )
