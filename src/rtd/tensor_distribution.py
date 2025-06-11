from abc import abstractmethod
from typing import Any, Dict

from torch import Size, Tensor
from torch.distributions import (
    Distribution,
    Independent,
    Normal,
    register_kl,
    kl_divergence,
)
import torch
from rtd.tensor_dict import TensorDict
from rtd.utils import PytreeRegistered


class TensorDistribution(TensorDict, PytreeRegistered):
    meta_data: Dict[str, Any]

    def __init__(self, data, shape, device, meta_data):
        super().__init__(data, shape, device)

        self.meta_data = meta_data

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


class TensorNormal(TensorDistribution):
    def __init__(
        self,
        loc,
        scale,
        reinterpreted_batch_ndims,
        shape=...,
        device=torch.device("cpu"),
    ):
        super().__init__(
            {"loc": loc, "scale": scale},
            shape,
            device,
            {"reinterpreted_batch_ndims": reinterpreted_batch_ndims},
        )

    def dist(self):
        return Independent(
            Normal(
                loc=self["loc"],
                scale=self["scale"],
            ),
            self.meta_data["reinterpreted_batch_ndims"],
        )


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
