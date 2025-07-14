from __future__ import annotations

from typing import Any, Dict, Optional, cast

from torch import Tensor
from torch.distributions import Chi2

from tensorcontainer.tensor_annotated import TDCompatible

from .base import TensorDistribution


class TensorChi2(TensorDistribution):
    """Tensor-aware Chi2 distribution.
    
    Creates a Chi2 distribution parameterized by degrees of freedom `df`.
    The Chi2 distribution is a continuous probability distribution that is
    a special case of the gamma distribution.
    
    Args:
        df: Degrees of freedom parameter. Must be positive.
        
    Note:
        The Chi2 distribution is commonly used in statistical hypothesis testing
        and confidence interval estimation.
    """
    
    # Annotated tensor parameters
    _df: Optional[Tensor] = None

    def __init__(self, df: Tensor):
        # Parameter validation occurs in super().__init__(), but we need an early
        # check here to safely derive shape and device from the data tensor
        # before calling the parent constructor
        if df is None:
            raise RuntimeError("'df' must be provided.")
        
        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        self._df = df
        
        shape = df.shape
        device = df.device

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        tensor_attributes: Dict[str, TDCompatible],
        meta_attributes: Dict[str, Any],
    ) -> TensorChi2:
        """Reconstruct distribution from tensor attributes."""
        df_param = tensor_attributes.get("_df")
        if df_param is None:
            raise RuntimeError("Cannot reconstruct TensorChi2: 'df' parameter is missing.")
        return cls(
            df=cast(Tensor, df_param),
        )

    def dist(self) -> Chi2:
        return Chi2(df=self._df)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def df(self) -> Tensor:
        """Returns the degrees of freedom parameter of the distribution."""
        return self.dist().df

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the Chi2 distribution."""
        return self.dist().variance

    @property
    def mean(self) -> Tensor:
        """Returns the mean of the Chi2 distribution."""
        return self.dist().mean
