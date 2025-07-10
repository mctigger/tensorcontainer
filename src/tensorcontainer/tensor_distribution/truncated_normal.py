from __future__ import annotations

from torch import Tensor
from torch.distributions import Distribution, Independent

from tensorcontainer.distributions.truncated_normal import TruncatedNormal
from .base import TensorDistribution


class TensorTruncatedNormal(TensorDistribution):
    loc: Tensor
    scale: Tensor
    low: Tensor
    high: Tensor
    reinterpreted_batch_ndims: int = 1

    def dist(self) -> Distribution:
        return Independent(
            TruncatedNormal(
                self.loc.float(),
                self.scale.float(),
                self.low.float(),  # type: ignore
                self.high.float(),  # type: ignore
            ),
            self.reinterpreted_batch_ndims,
        )
