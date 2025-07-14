from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Distribution, Independent

from tensorcontainer.distributions.truncated_normal import TruncatedNormal

from .base import TensorDistribution


class TensorTruncatedNormal(TensorDistribution):
    loc: Tensor
    scale: Tensor
    low: Tensor
    high: Tensor
    def __init__(
        self,
        loc: Tensor,
        scale: Tensor,
        low: Tensor,
        high: Tensor,
        shape: torch.Size,
        device: torch.device,
    ):
        self.loc = loc
        self.scale = scale
        self.low = low
        self.high = high
        super().__init__(shape=shape, device=device)
        self.high = high

    def dist(self) -> Distribution:
        return Independent(
            TruncatedNormal(
                self.loc.float(),
                self.scale.float(),
                self.low.float(),  # type: ignore
                self.high.float(),  # type: ignore
            ),
            1,
        )
