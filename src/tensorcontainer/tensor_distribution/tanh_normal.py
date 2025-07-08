from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import (
    Distribution,
    Independent,
    Normal,
    TransformedDistribution,
    constraints,
)

from tensorcontainer.distributions.sampling import SamplingDistribution
from .base import TensorDistribution


class ClampedTanhTransform(torch.distributions.transforms.Transform):
    """
    Transform that applies tanh and clamps the output between -1 and 1.
    """

    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True

    @property
    def sign(self):
        return +1

    def __init__(self):
        super().__init__()

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        # Arctanh
        return torch.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # |det J| = 1 - tanh^2(x)
        # log|det J| = log(1 - tanh^2(x))
        return torch.log(
            1 - y.pow(2) + 1e-6
        )  # Adding small epsilon for numerical stability


class TensorTanhNormal(TensorDistribution):
    loc: Tensor
    scale: Tensor
    reinterpreted_batch_ndims: int = 1

    def __post_init__(self):
        super().__post_init__()

    def dist(self) -> Distribution:
        return Independent(
            SamplingDistribution(
                TransformedDistribution(
                    Normal(self.loc.float(), self.scale.float()),
                    [
                        ClampedTanhTransform(),
                    ],
                ),
            ),
            self.reinterpreted_batch_ndims,
        )

    def copy(self):
        return TensorTanhNormal(
            loc=self.loc,
            scale=self.scale,
            reinterpreted_batch_ndims=self.reinterpreted_batch_ndims,
            shape=self.shape,
            device=self.device,
        )
