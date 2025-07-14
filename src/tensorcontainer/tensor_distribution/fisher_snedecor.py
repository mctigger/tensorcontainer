from __future__ import annotations

from typing import Any, Dict

from torch import Size, Tensor
from torch.distributions import FisherSnedecor as TorchFisherSnedecor

from tensorcontainer.tensor_annotated import TDCompatible

from .base import TensorDistribution


class FisherSnedecor(TensorDistribution):
    """
    A Fisher-Snedecor distribution.

    This distribution is parameterized by two degrees of freedom parameters, `df1` and `df2`.

    Source: https://pytorch.org/docs/stable/distributions.html#fishersnedecor
    """

    # Annotated tensor parameters
    _df1: Tensor
    _df2: Tensor

    def __init__(self, df1: Tensor, df2: Tensor):
        # Parameter validation occurs in super().__init__(), but we need an early
        # check here to safely derive shape and device from the data tensor
        # before calling the parent constructor
        if df1 is None or df2 is None:
            raise RuntimeError("Both 'df1' and 'df2' must be provided.")

        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        self._df1 = df1
        self._df2 = df2

        shape = self._df1.shape
        device = self._df1.device

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        tensor_attributes: Dict[str, TDCompatible],
        meta_attributes: Dict[str, Any],
    ) -> FisherSnedecor:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            df1=tensor_attributes["_df1"],  # type: ignore
            df2=tensor_attributes["_df2"],  # type: ignore
        )

    def dist(self) -> TorchFisherSnedecor:
        return TorchFisherSnedecor(df1=self._df1, df2=self._df2)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def df1(self) -> Tensor:
        """Returns the first degree of freedom parameter."""
        return self.dist().df1

    @property
    def df2(self) -> Tensor:
        """Returns the second degree of freedom parameter."""
        return self.dist().df2

    @property
    def param_shape(self) -> Size:
        """Returns the shape of the underlying parameter."""
        return self.dist().batch_shape
