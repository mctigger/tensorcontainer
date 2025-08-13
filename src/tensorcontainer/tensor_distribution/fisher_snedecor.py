from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.distributions import FisherSnedecor as TorchFisherSnedecor
from .utils import broadcast_all

from .base import TensorDistribution


class TensorFisherSnedecor(TensorDistribution):
    def __init__(
        self, df1: Tensor, df2: Tensor, *, validate_args: bool | None = None
    ) -> None:
        self._df1: Tensor
        self._df2: Tensor
        self._df1, self._df2 = broadcast_all(df1, df2)
        batch_shape = self._df1.shape
        super().__init__(batch_shape, self._df1.device, validate_args)

    def dist(self) -> TorchFisherSnedecor:
        return TorchFisherSnedecor(
            self._df1, self._df2, validate_args=self._validate_args
        )

    @classmethod
    def _unflatten_distribution(
        cls, attributes: dict[str, Any]
    ) -> TensorFisherSnedecor:
        return cls(
            df1=attributes["_df1"],
            df2=attributes["_df2"],
            validate_args=attributes.get("_validate_args"),
        )
