from __future__ import annotations

import dataclasses
from abc import abstractmethod

import torch
from torch import Size, Tensor
from torch.distributions import Distribution, kl_divergence, register_kl

from tensorcontainer.tensor_dataclass import TensorDataClass


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

        # Instantiate the torch.distributions.Distribution fail early if
        # constraints are not uphold
        self.dist()

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
