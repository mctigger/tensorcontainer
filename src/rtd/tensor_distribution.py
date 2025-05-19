from abc import ABC, abstractmethod
from typing import Any, Dict

from torch import Size, Tensor
from torch.distributions import (
    Distribution,
    Independent,
    Normal,
    Bernoulli,
    register_kl,
    kl_divergence,
)
import torch
from rtd.tensor_dict import TensorDict


class SoftBernoulli(Bernoulli):
    def log_prob(self, value):
        # Compute log probabilities
        log_p = super().log_prob(torch.ones_like(value))  # log(p)
        log_1_p = super().log_prob(torch.zeros_like(value))  # log(1 - p)

        # Compute soft BCE using the original formula
        log_probs = value * log_p + (1 - value) * log_1_p

        return log_probs


class TensorDistribution(TensorDict, ABC):
    distribution_properties: Dict[str, Any]

    def __init__(self, data, shape, device, distribution_properties):
        super().__init__(data, shape, device)

        self.distribution_properties = distribution_properties

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

    def copy(self):
        td = super().copy()
        cls = type(self)
        obj = cls.__new__(cls)
        TensorDistribution.__init__(
            obj,
            td.data,
            td.shape,
            td.device,
            self.distribution_properties,
        )
        return obj

    def apply(self, fn):
        td = super().apply(fn)

        cls = type(self)
        obj = cls.__new__(cls)
        TensorDistribution.__init__(
            obj, td.data, td.shape, td.device, self.distribution_properties
        )

        return obj

    @classmethod
    def zip_apply(cls, tensor_dicts, fn):
        td = TensorDict.zip_apply(tensor_dicts, fn)

        obj = cls.__new__(cls)
        TensorDistribution.__init__(
            obj, td.data, td.shape, td.device, tensor_dicts[0].distribution_properties
        )

        return obj


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
            self.distribution_properties["reinterpreted_batch_ndims"],
        )


class TensorBernoulli(TensorDistribution):
    def __init__(
        self,
        logits=None,
        probs=None,
        soft: bool = False,
        reinterpreted_batch_ndims=1,
        shape=...,
        device=torch.device("cpu"),
    ):
        if logits is None and probs is None:
            raise ValueError("Either logits or probs must be provided.")
        if logits is not None and probs is not None:
            raise ValueError("Only one of logits or probs should be provided.")

        data = {}
        if logits is not None:
            data["logits"] = logits
        if probs is not None:
            data["probs"] = probs

        super().__init__(
            data,
            shape,
            device,
            {"reinterpreted_batch_ndims": reinterpreted_batch_ndims, "soft": soft},
        )

    def dist(self) -> Distribution:
        kwargs = {}
        if "logits" in self:
            kwargs["logits"] = self["logits"]
        if "probs" in self:
            kwargs["probs"] = self["probs"]
        if self.distribution_properties["soft"]:
            return Independent(
                SoftBernoulli(**kwargs),
                self.distribution_properties["reinterpreted_batch_ndims"],
            )
        else:
            return Independent(
                Bernoulli(**kwargs),
                self.distribution_properties["reinterpreted_batch_ndims"],
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
