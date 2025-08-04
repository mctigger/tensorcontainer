from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.distributions import StudentT
from tensorcontainer.tensor_distribution.utils import broadcast_all


from .base import TensorDistribution


class TensorStudentT(TensorDistribution):
    """Tensor-aware Student's t-distribution.

    Creates a Student's t-distribution parameterized by degrees of freedom `df`,
    mean `loc` and scale `scale`.

    Args:
        df: Degrees of freedom. Must be positive.
        loc: Mean of the distribution (default: 0.0).
        scale: Scale of the distribution. Must be positive (default: 1.0).
        validate_args: Whether to validate arguments (default: None).

    Note:
        The Student's t-distribution approaches the standard normal distribution
        as the degrees of freedom approaches infinity.
    """

    # Annotated tensor parameters
    _df: Tensor
    _loc: Tensor
    _scale: Tensor

    def __init__(
        self,
        df: Tensor | float,
        loc: Tensor | float = 0.0,
        scale: Tensor | float = 1.0,
        validate_args: bool | None = None,
    ) -> None:
        self._df, self._loc, self._scale = broadcast_all(df, loc, scale)

        super().__init__(self._df.shape, self._df.device, validate_args)

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: dict[str, Any],
    ) -> "TensorStudentT":
        return cls(
            df=attributes["_df"],
            loc=attributes["_loc"],
            scale=attributes["_scale"],
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> StudentT:
        return StudentT(
            df=self._df,
            loc=self._loc,
            scale=self._scale,
            validate_args=self._validate_args,
        )

    @property
    def df(self) -> Tensor:
        """Returns the degrees of freedom of the StudentT distribution."""
        return self.dist().df

    @property
    def loc(self) -> Tensor:
        """Returns the mean of the StudentT distribution."""
        return self.dist().loc

    @property
    def scale(self) -> Tensor:
        """Returns the scale of the StudentT distribution."""
        return self.dist().scale
