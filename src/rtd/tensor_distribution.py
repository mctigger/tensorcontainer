from abc import ABC, abstractmethod
from typing import Dict, Any
from rtd.tensor_dict import TensorDict
from torch.distributions import Independent, Normal


class TensorDistribution(TensorDict, ABC):
    distribution_properties: Dict[str, Any]

    def __init__(self, data, shape, distribution_properties):
        super().__init__(data, shape)

        self.distribution_properties = distribution_properties

    @abstractmethod
    def dist(self): ...

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
