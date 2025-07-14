from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.distributions import StudentT

from tensorcontainer.tensor_annotated import TDCompatible

from .base import TensorDistribution


class TensorStudentT(TensorDistribution):
    """Tensor-aware StudentT distribution."""

    # Annotated tensor parameters
    _df: Optional[Tensor] = None
    _loc: Optional[Tensor] = None
    _scale: Optional[Tensor] = None

    def __init__(
        self,
        df: Tensor,
        loc: float = 0.0,
        scale: float = 1.0,
    ):
        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        self._df = df
        self._loc = torch.as_tensor(loc, dtype=df.dtype, device=df.device)
        self._scale = torch.as_tensor(scale, dtype=df.dtype, device=df.device)

        if torch.any(self._df <= 0):
            raise ValueError("df must be positive")
        if torch.any(self._scale <= 0):
            raise ValueError("scale must be positive")

        try:
            batch_shape = torch.broadcast_shapes(self._df.shape, self._loc.shape, self._scale.shape)
        except RuntimeError as e:
            raise ValueError(f"df, loc, and scale must have compatible shapes: {e}")

        super().__init__(batch_shape, self._df.device)

    @classmethod
    def _unflatten_distribution(
        cls,
        tensor_attributes: Dict[str, TDCompatible],
        meta_attributes: Dict[str, Any],
    ) -> "TensorStudentT":
        """Reconstruct distribution from tensor attributes."""
        return cls(
            df=torch.as_tensor(tensor_attributes["_df"]),
            loc=float(torch.as_tensor(tensor_attributes["_loc"])),
            scale=float(torch.as_tensor(tensor_attributes["_scale"])),
        )

    def dist(self) -> StudentT:
        assert self._df is not None
        assert self._loc is not None
        assert self._scale is not None
        return StudentT(df=self._df, loc=self._loc, scale=self._scale)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def df(self) -> Optional[Tensor]:
        """Returns the degrees of freedom of the StudentT distribution."""
        return self.dist().df

    @property
    def loc(self) -> Optional[Tensor]:
        """Returns the mean of the StudentT distribution."""
        return self.dist().loc

    @property
    def scale(self) -> Optional[Tensor]:
        """Returns the scale of the StudentT distribution."""
        return self.dist().scale

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the StudentT distribution."""
        assert self._df is not None
        assert self._scale is not None
        var = torch.full_like(self._df, float("inf"))
        var[self._df > 2] = (self._scale**2 * (self._df / (self._df - 2)))[self._df > 2]
        var[self._df <= 1] = float("nan")
        return var
