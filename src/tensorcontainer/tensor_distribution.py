from __future__ import annotations

import dataclasses
from abc import abstractmethod
from dataclasses import field
from typing import Optional

import torch
from torch import Size, Tensor
from torch.distributions import (
    Distribution,
    Independent,
    Normal,
    OneHotCategoricalStraightThrough,
    TransformedDistribution,
    constraints,
    kl_divergence,
    register_kl,
)

from tensorcontainer.distributions.sampling import SamplingDistribution
from tensorcontainer.distributions.soft_bernoulli import SoftBernoulli
from tensorcontainer.distributions.truncated_normal import TruncatedNormal
from tensorcontainer.tensor_dataclass import TensorDataClass


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


class TensorDistribution(TensorDataClass):
    def __post_init__(self):
        # infer shape and device if not provided
        for f in dataclasses.fields(self):
            if f.name in ("shape", "device"):
                continue
            val = getattr(self, f.name)
            if isinstance(val, Tensor):
                if self.shape is None:
                    self.shape = torch.Size(val.shape)
                if self.device is None:
                    self.device = val.device
                break

        super().__post_init__()

    @abstractmethod
    def dist(self) -> Distribution:
        """Returns the underlying torch.distributions.Distribution instance."""

    def rsample(self, sample_shape: Size = Size()) -> Tensor:
        return self.dist().rsample(sample_shape)

    def sample(self, sample_shape: Size = Size()) -> Tensor:
        return self.dist().sample(sample_shape)

    def entropy(self) -> Tensor:
        return self.dist().entropy()

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def mean(self) -> Tensor:
        return self.dist().mean

    @property
    def stddev(self) -> Tensor:
        return self.dist().stddev

    @property
    def mode(self) -> Tensor:
        return self.dist().mode


class TensorNormal(TensorDistribution):
    loc: Tensor
    scale: Tensor
    reinterpreted_batch_ndims: int = 1

    def dist(self) -> Distribution:
        return Independent(
            Normal(
                loc=self.loc,
                scale=self.scale,
            ),
            self.reinterpreted_batch_ndims,
        )


class TensorTruncatedNormal(TensorDistribution):
    loc: Tensor
    scale: Tensor
    low: Tensor
    high: Tensor
    reinterpreted_batch_ndims: int = 1

    def __post_init__(self):
        super().__post_init__()

    def dist(self) -> Distribution:
        return Independent(
            TruncatedNormal(
                self.loc.float(),
                self.scale.float(),
                self.low.float(),  # type: ignore
                self.high.float(),  # type: ignore
            ),
            self.reinterpreted_batch_ndims,
        )


class TensorBernoulli(TensorDistribution):
    _probs: Optional[Tensor] = field(default=None)
    _logits: Optional[Tensor] = field(default=None)
    reinterpreted_batch_ndims: int = 0

    def __post_init__(self):
        super().__post_init__()  # Call parent's post_init
        if (self._probs is None) == (self._logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )

    @property
    def probs(self):
        if self._probs is None:
            assert self._logits is not None
            self._probs = torch.sigmoid(self._logits)
        return self._probs

    @probs.setter
    def probs(self, value):
        self._probs = value
        if value is not None:
            self._logits = None

    @property
    def logits(self):
        if self._logits is None:
            assert self._probs is not None
            self._logits = torch.log(self._probs / (1 - self._probs + 1e-8))
        return self._logits

    @logits.setter
    def logits(self, value):
        self._logits = value
        if value is not None:
            self._probs = None

    def dist(self):
        if self._probs is not None:
            return Independent(
                torch.distributions.Bernoulli(
                    probs=self._probs,
                ),
                self.reinterpreted_batch_ndims,
            )
        else:
            return Independent(
                torch.distributions.Bernoulli(
                    logits=self._logits,
                ),
                self.reinterpreted_batch_ndims,
            )


class TensorSoftBernoulli(TensorDistribution):
    _probs: Optional[Tensor] = field(default=None)
    _logits: Optional[Tensor] = field(default=None)
    reinterpreted_batch_ndims: int = 0

    def __post_init__(self):
        super().__post_init__()  # Call parent's post_init
        if (self._probs is None) == (self._logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )

    @property
    def probs(self):
        if self._probs is None:
            assert self._logits is not None
            self._probs = torch.sigmoid(self._logits)
        return self._probs

    @probs.setter
    def probs(self, value):
        self._probs = value
        if value is not None:
            self._logits = None

    @property
    def logits(self):
        if self._logits is None:
            assert self._probs is not None
            self._logits = torch.log(self._probs / (1 - self._probs + 1e-8))
        return self._logits

    @logits.setter
    def logits(self, value):
        self._logits = value
        if value is not None:
            self._probs = None

    def dist(self):
        if self._probs is not None:
            return Independent(
                SoftBernoulli(
                    probs=self._probs,
                ),
                self.reinterpreted_batch_ndims,
            )
        else:
            return Independent(
                SoftBernoulli(
                    logits=self._logits,
                ),
                self.reinterpreted_batch_ndims,
            )


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


@register_kl(TensorDistribution, TensorDistribution)
def registerd_td_td(
    td_a: TensorDistribution,
    td_b: TensorDistribution,
):
    return kl_divergence(td_a.dist(), td_b.dist())


@register_kl(TensorDistribution, Distribution)
def register_td_d(td: TensorDistribution, d: Distribution):
    return kl_divergence(td.dist(), d)


@register_kl(Distribution, TensorDistribution)
def registerd_d_td(
    d: Distribution,
    td: TensorDistribution,
):
    return kl_divergence(d, td.dist())


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
