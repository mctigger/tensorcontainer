from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.distributions import Chi2 as TorchChi2
from .utils import broadcast_all

from tensorcontainer.tensor_distribution.base import TensorDistribution


class TensorChi2(TensorDistribution):
    r"""
    Creates a Chi-squared distribution parameterized by shape parameter :attr:`df`.
    This is exactly equivalent to ``Gamma(alpha=0.5*df, beta=0.5)``

    Args:
        df (float or Tensor): shape parameter of the distribution
    """

    _df: Tensor

    def __init__(self, df: float | Tensor, validate_args: bool | None = None) -> None:
        # Use broadcast_all to handle Union[float, Tensor] and ensure tensor conversion
        (self._df,) = broadcast_all(df)

        super().__init__(self._df.shape, self._df.device, validate_args)

    def dist(self) -> TorchChi2:
        return TorchChi2(df=self._df, validate_args=self._validate_args)

    @classmethod
    def _unflatten_distribution(cls, attributes: dict[str, Any]) -> "TensorChi2":
        return cls(df=attributes["_df"], validate_args=attributes.get("_validate_args"))

    @property
    def df(self) -> Tensor:
        return self.dist().df
