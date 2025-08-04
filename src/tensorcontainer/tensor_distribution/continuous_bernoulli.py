from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.distributions import ContinuousBernoulli as TorchContinuousBernoulli
from .utils import broadcast_all
from torch.types import Number

from .base import TensorDistribution


class TensorContinuousBernoulli(TensorDistribution):
    _probs: Tensor | None
    _logits: Tensor | None
    _lims: tuple[float, float]

    def __init__(
        self,
        probs: Tensor | Number | None = None,
        logits: Tensor | Number | None = None,
        lims: tuple[float, float] = (0.499, 0.501),
        validate_args: bool | None = None,
    ) -> None:
        self._lims = lims
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )

        if probs is not None:
            (self._probs,) = broadcast_all(probs)
            self._logits = None
            data = self._probs
        else:
            (self._logits,) = broadcast_all(logits)
            self._probs = None
            data = self._logits

        batch_shape = data.shape  # type: ignore
        device = data.device  # type: ignore

        super().__init__(shape=batch_shape, device=device, validate_args=validate_args)

    def dist(self) -> TorchContinuousBernoulli:
        return TorchContinuousBernoulli(
            probs=self._probs,
            logits=self._logits,
            lims=self._lims,
            validate_args=self._validate_args,
        )

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: dict[str, Any],
    ) -> "TensorContinuousBernoulli":
        return cls(
            probs=attributes.get("_probs"),
            logits=attributes.get("_logits"),
            lims=attributes["_lims"],
            validate_args=attributes.get("_validate_args"),
        )

    @property
    def probs(self) -> Tensor:
        return self.dist().probs

    @property
    def logits(self) -> Tensor:
        return self.dist().logits

    @property
    def param_shape(self) -> torch.Size:
        return self.dist().param_shape
