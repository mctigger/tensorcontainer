from __future__ import annotations

from torch import Tensor
from torch.distributions import (
    Distribution,
    Independent,
    OneHotCategoricalStraightThrough,
)

from .base import TensorDistribution


class TensorCategorical(TensorDistribution):
    logits: Tensor
    reinterpreted_batch_ndims: int = 1

    def __post_init__(self):
        super().__post_init__()

    def dist(self) -> Distribution:
        one_hot = OneHotCategoricalStraightThrough(logits=self.logits.float())
        if self.reinterpreted_batch_ndims < 1:
            raise ValueError(
                "reinterpreted_batch_ndims must be at least 1 for TensorCategorical, "
                f"but got {self.reinterpreted_batch_ndims}"
            )
        dims_to_reinterpret = self.reinterpreted_batch_ndims - 1
        return Independent(one_hot, dims_to_reinterpret)

    def entropy(self) -> Tensor:
        return OneHotCategoricalStraightThrough(logits=self.logits.float()).entropy()

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)
