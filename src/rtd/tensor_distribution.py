from __future__ import annotations

from typing import Optional
from abc import abstractmethod
from dataclasses import field

from torch import Size, Tensor
from torch.distributions import (
    Distribution,
    Independent,
    Normal,
    TransformedDistribution,
    register_kl,
    kl_divergence,
    OneHotCategoricalStraightThrough,
)
from rtd.distributions.sampling import SamplingDistribution
import torch
from rtd.distributions.soft_bernoulli import SoftBernoulli
from rtd.distributions.truncated_normal import TruncatedNormal
from rtd.tensor_dataclass import TensorDataclass


class ClampedTanhTransform(torch.distributions.transforms.Transform):
    """
    Transform that applies tanh and clamps the output between -1 and 1.
    """

    domain = torch.distributions.constraints.real
    codomain = torch.distributions.constraints.interval(-1.0, 1.0)
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


class TensorDistribution(TensorDataclass):
    @abstractmethod
    def dist(self) -> Distribution: ...

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


class TensorNormal(TensorDataclass):
    loc: Tensor
    scale: Tensor
    reinterpreted_batch_ndims: int = 1
    shape: tuple = field(default_factory=tuple, init=False)
    device: Optional[torch.device] = field(default=None, init=False)

    def __post_init__(self):
        super().__post_init__()

    def dist(self) -> Distribution:
        return Independent(
            Normal(
                loc=self.loc,
                scale=self.scale,
            ),
            self.reinterpreted_batch_ndims,
        )

    def copy(self):
        return TensorNormal(
            loc=self.loc.clone(),
            scale=self.scale.clone(),
            reinterpreted_batch_ndims=self.reinterpreted_batch_ndims,
            shape=self.shape,
            device=self.device,
        )


class TensorTruncatedNormal(TensorDataclass):
    loc: Tensor
    scale: Tensor
    low: Tensor
    high: Tensor
    reinterpreted_batch_ndims: int = 1  # Default value for reinterpreted_batch_ndims
    shape: tuple = field(default_factory=tuple, init=False)
    device: Optional[torch.device] = field(default=None, init=False)

    def __post_init__(self):
        super().__post_init__()

    def dist(self) -> Distribution:
        return Independent(
            TruncatedNormal(
                self.loc.float(),
                self.scale.float(),
                self.low,
                self.high,
            ),
            self.reinterpreted_batch_ndims,
        )

    def copy(self):
        return TensorTruncatedNormal(
            loc=self.loc.clone(),
            scale=self.scale.clone(),
            low=self.low.clone(),
            high=self.high.clone(),
            reinterpreted_batch_ndims=self.reinterpreted_batch_ndims,
            shape=self.shape,
            device=self.device,
        )


class TensorBernoulli(TensorDataclass):
    probs: Optional[Tensor] = field(default=None)
    logits: Optional[Tensor] = field(default=None)
    reinterpreted_batch_ndims: int = 0
    shape: tuple = field(default_factory=tuple, init=False)
    device: Optional[torch.device] = field(default=None, init=False)

    def __post_init__(self):
        super().__post_init__()  # Call parent's post_init
        if (self.probs is None) == (self.logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        # Ensure _probs and _logits are initialized for properties
        if self.probs is not None:
            self._probs = self.probs
            self._logits = None
        elif self.logits is not None:
            self._logits = self.logits
            self._probs = None

    @property
    def probs(self):
        if self._probs is None and self._logits is not None:
            self._probs = torch.sigmoid(self._logits)
        return self._probs

    @probs.setter
    def probs(self, value):
        self._probs = value
        self._logits = None  # Invalidate logits when probs is set

    @property
    def logits(self):
        if self._logits is None and self._probs is not None:
            # Add a small epsilon to avoid log(0) or log(negative)
            self._logits = torch.log(self._probs / (1 - self._probs + 1e-8))
        return self._logits

    @logits.setter
    def logits(self, value):
        self._logits = value
        self._probs = None  # Invalidate probs when logits is set

    def dist(self):
        if self.probs is not None:
            return Independent(
                torch.distributions.Bernoulli(
                    probs=self.probs,
                ),
                self.reinterpreted_batch_ndims,
            )
        else:
            return Independent(
                torch.distributions.Bernoulli(
                    logits=self.logits,
                ),
                self.reinterpreted_batch_ndims,
            )

    def copy(self):
        if self.probs is not None:
            return TensorBernoulli(
                probs=self.probs.clone(),
                reinterpreted_batch_ndims=self.reinterpreted_batch_ndims,
                shape=self.shape,
                device=self.device,
            )
        else:
            return TensorBernoulli(
                logits=self.logits.clone(),
                reinterpreted_batch_ndims=self.reinterpreted_batch_ndims,
                shape=self.shape,
                device=self.device,
            )


class TensorSoftBernoulli(TensorDataclass):
    probs: Optional[Tensor] = field(default=None)
    logits: Optional[Tensor] = field(default=None)
    reinterpreted_batch_ndims: int = 0
    shape: tuple = field(default_factory=tuple, init=False)
    device: Optional[torch.device] = field(default=None, init=False)

    def __post_init__(self):
        super().__post_init__()  # Call parent's post_init
        if (self.probs is None) == (self.logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        # Ensure _probs and _logits are initialized for properties
        if self.probs is not None:
            self._probs = self.probs
            self._logits = None
        elif self.logits is not None:
            self._logits = self.logits
            self._probs = None

    @property
    def probs(self):
        if self._probs is None and self._logits is not None:
            self._probs = torch.sigmoid(self._logits)
        return self._probs

    @probs.setter
    def probs(self, value):
        self._probs = value
        self._logits = None  # Invalidate logits when probs is set

    @property
    def logits(self):
        if self._logits is None and self._probs is not None:
            # Add a small epsilon to avoid log(0) or log(negative)
            self._logits = torch.log(self._probs / (1 - self._probs + 1e-8))
        return self._logits

    @logits.setter
    def logits(self, value):
        self._logits = value
        self._probs = None  # Invalidate probs when logits is set

    def dist(self):
        if self.probs is not None:
            return Independent(
                SoftBernoulli(
                    probs=self.probs,
                ),
                self.reinterpreted_batch_ndims,
            )
        else:
            return Independent(
                SoftBernoulli(
                    logits=self.logits,
                ),
                self.reinterpreted_batch_ndims,
            )

    def copy(self):
        if self.probs is not None:
            return TensorSoftBernoulli(
                probs=self.probs.clone(),
                reinterpreted_batch_ndims=self.reinterpreted_batch_ndims,
                shape=self.shape,
                device=self.device,
            )
        else:
            return TensorSoftBernoulli(
                logits=self.logits.clone(),
                reinterpreted_batch_ndims=self.reinterpreted_batch_ndims,
                shape=self.shape,
                device=self.device,
            )


class TensorCategorical(TensorDataclass):
    logits: Tensor
    output_shape: tuple
    reinterpreted_batch_ndims: int = len
    shape: tuple = field(default_factory=tuple, init=False)
    device: Optional[torch.device] = field(default=None, init=False)

    def __post_init__(self):
        super().__post_init__()

    def dist(self):
        logits = self.logits.float()
        output_shape = self.output_shape
        logits = logits.view(*logits.shape[:-1], -1, *output_shape)
        one_hot = OneHotCategoricalStraightThrough(logits=logits)

        return Independent(one_hot, self.reinterpreted_batch_ndims)


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


class TensorTanhNormal(TensorDataclass):
    loc: Tensor
    scale: Tensor
    reinterpreted_batch_ndims: int = 1  # Default value for reinterpreted_batch_ndims
    shape: tuple = field(default_factory=tuple, init=False)
    device: Optional[torch.device] = field(default=None, init=False)

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
            loc=self.loc.clone(),
            scale=self.scale.clone(),
            reinterpreted_batch_ndims=self.reinterpreted_batch_ndims,
            shape=self.shape,
            device=self.device,
        )
