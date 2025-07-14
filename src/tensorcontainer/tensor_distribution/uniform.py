from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.distributions import Uniform

from tensorcontainer.tensor_annotated import TDCompatible

from .base import TensorDistribution


class TensorUniform(TensorDistribution):
    _low: Optional[Tensor] = None
    _high: Optional[Tensor] = None

    def __init__(self, low: Tensor, high: Tensor):
        if low is None:
            raise RuntimeError("'low' must be provided.")
        if high is None:
            raise RuntimeError("'high' must be provided.")

        batch_shape = torch.broadcast_shapes(low.shape, high.shape)
        device = low.device if low.is_cuda else high.device if high.is_cuda else low.device
        self._low = low
        self._high = high

        super().__init__(batch_shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        tensor_attributes: Dict[str, TDCompatible],
        meta_attributes: Dict[str, Any],
    ) -> TensorUniform:
        return cls(
            low=tensor_attributes.get("_low"),  # type: ignore
            high=tensor_attributes.get("_high"),  # type: ignore
        )

    def dist(self) -> Uniform:
        return Uniform(low=self._low, high=self._high)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def low(self) -> Tensor:
        return self.dist().low

    @property
    def high(self) -> Tensor:
        return self.dist().high

    @property
    def mean(self) -> Tensor:
        return self.dist().mean

    @property
    def variance(self) -> Tensor:
        return self.dist().variance

    @property
    def stddev(self) -> Tensor:
        return self.dist().stddev
