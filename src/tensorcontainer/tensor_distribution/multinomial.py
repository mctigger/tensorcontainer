from __future__ import annotations

from typing import Optional

from torch import Tensor
from torch.distributions import Distribution, Independent, Multinomial

from .base import TensorDistribution


class TensorMultinomial(TensorDistribution):
    total_count: int
    probs: Optional[Tensor] = None
    logits: Optional[Tensor] = None
    reinterpreted_batch_ndims: int = 1

    def __post_init__(self):
        super().__post_init__()

        if self.total_count < 1:
            raise ValueError("total_count must be a positive integer")

        if self.probs is None and self.logits is None:
            raise ValueError("Either probs or logits must be specified")
        if self.probs is not None and self.logits is not None:
            raise ValueError("Only one of probs or logits can be specified")

    def dist(self) -> Distribution:
        base_dist = Multinomial(
            total_count=self.total_count,
            probs=self.probs,
            logits=self.logits,
            validate_args=False,
        )

        # Only use Independent if reinterpreted_batch_ndims > 0 and we have batch dimensions
        if (
            self.reinterpreted_batch_ndims > 0
            and len(base_dist.batch_shape) >= self.reinterpreted_batch_ndims
        ):
            return Independent(base_dist, self.reinterpreted_batch_ndims)
        else:
            return base_dist

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the Multinomial distribution."""
        return self.dist().variance
