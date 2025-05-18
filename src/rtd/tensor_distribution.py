from abc import ABC, abstractmethod
from typing import Any, Dict

from torch import Size, Tensor
from torch.distributions import Distribution, Independent, Normal

from rtd.tensor_dict import TensorDict


class TensorDistribution(TensorDict, ABC):
    distribution_properties: Dict[str, Any]

    def __init__(self, data, shape, distribution_properties):
        super().__init__(data, shape)

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

    def apply(self, fn):
        td = super().apply(fn)

        cls = type(self)
        obj = cls.__new__(cls)
        TensorDistribution.__init__(
            obj, td.data, td.shape, self.distribution_properties
        )

        return obj

    @classmethod
    def zip_apply(cls, tensor_dicts, fn):
        td = TensorDict.zip_apply(tensor_dicts, fn)

        obj = cls.__new__(cls)
        TensorDistribution.__init__(
            obj, td.data, td.shape, tensor_dicts[0].distribution_properties
        )

        return obj


class TensorNormal(TensorDistribution):
    def __init__(self, loc, scale, reinterpreted_batch_ndims, shape=...):
        super().__init__(
            {"loc": loc, "scale": scale},
            shape,
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
