from __future__ import annotations


from torch.distributions import (
    OneHotCategoricalStraightThrough as TorchOneHotCategoricalStraightThrough,
)

from .one_hot_categorical import TensorOneHotCategorical


class TensorOneHotCategoricalStraightThrough(TensorOneHotCategorical):
    """Tensor-aware OneHotCategoricalStraightThrough distribution."""

    def dist(self) -> TorchOneHotCategoricalStraightThrough:
        return TorchOneHotCategoricalStraightThrough(
            probs=self._probs, logits=self._logits, validate_args=self._validate_args
        )
