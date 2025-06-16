from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict

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
from rtd.tensor_dict import TensorDict
from rtd.utils import PytreeRegistered

from typing import List, Tuple


# Use the official PyTree utility from torch
import torch.utils._pytree as pytree


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


class TensorDistribution(TensorDict, PytreeRegistered):
    meta_data: Dict[str, Any]

    def __init__(self, data, shape, device, meta_data):
        super().__init__(data, shape, device)

        self.meta_data = meta_data

    def _get_pytree_context(
        self, flat_leaves: List[Tensor], children_spec: pytree.TreeSpec
    ) -> Tuple:
        """
        Private helper to compute the pytree context for this TensorDict.

        The context captures the necessary metadata to reconstruct the TensorDict
        from its leaves: the original structure of the contained data and the
        event dimensions of each tensor.
        """
        batch_ndim = len(self.shape)
        event_ndims = tuple(leaf.ndim - batch_ndim for leaf in flat_leaves)
        return (children_spec, event_ndims, self.meta_data)

    def _pytree_flatten(self) -> Tuple[List[Tensor], Tuple]:
        """
        Flattens the TensorDict into its tensor leaves and static metadata.
        (Implementation for `flatten_fn` in `register_pytree_node`)
        """
        # Get the leaves and the spec describing the structure of self.data
        flat_leaves, children_spec = pytree.tree_flatten(self.data)

        # Use the helper to compute and return the context
        context = self._get_pytree_context(flat_leaves, children_spec)
        return flat_leaves, context

    def _pytree_flatten_with_keys_fn(
        self,
    ) -> Tuple[List[Tuple[pytree.KeyPath, Tensor]], Tuple]:
        """
        Flattens the TensorDict into key-path/leaf pairs and static metadata.
        (Implementation for `flatten_with_keys_fn` in `register_pytree_node`)
        """
        # Use the public API to robustly get key paths, leaves, and the spec
        keypath_leaf_list, children_spec = pytree.tree_flatten_with_path(self.data)

        # Extract just the leaves to pass to the context helper
        flat_leaves = [leaf for _, leaf in keypath_leaf_list]

        # Use the helper to compute and return the context
        context = self._get_pytree_context(flat_leaves, children_spec)
        return keypath_leaf_list, context

    @classmethod
    def _pytree_unflatten(cls, leaves: List[Tensor], context: Tuple) -> TensorDict:
        """
        Reconstructs a TensorDict by creating a new instance and manually
        populating its attributes. This approach is more robust for torch.compile's
        code generation phase.
        """
        (children_spec, event_ndims, meta_data) = context  # Unpack the context

        if not leaves:
            # Handle the empty case explicitly with direct instantiation
            obj = cls.__new__(cls)
            obj.data = {}
            obj.shape = []  # Or a sensible default for empty
            obj.device = None
            obj.meta_data = {}
            return obj

        # Reconstruct the nested dictionary structure using the unflattened leaves
        data = pytree.tree_unflatten(leaves, children_spec)

        # Infer new_shape and new_device
        first_leaf_reconstructed = leaves[0]

        # Simplified inference (common and works for stack/cat):
        new_device = first_leaf_reconstructed.device

        # Calculate new_shape based on the structure and first leaf.
        # For operations like `stack`, the batch shape changes.
        # If `_pytree_flatten` correctly passes `event_ndims`, then:
        if event_ndims[0] == 0:
            new_shape = first_leaf_reconstructed.shape
        else:
            new_shape = first_leaf_reconstructed.shape[: -event_ndims[0]]

        # Instead of calling `_reconstruct_tensordict` which wraps `cls(...)`,
        # directly use `cls.__new__` and set attributes.
        obj = cls.__new__(cls)
        obj.data = (
            data  # This is the reconstructed nested dictionary of tensors/TensorDicts
        )
        obj.shape = new_shape
        obj.device = new_device
        obj.meta_data = meta_data
        return obj

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data={self.data}, shape={self.shape}, device={self.device}, meta_data={self.meta_data})"

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
        shape,
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

    def copy(self):
        return TensorNormal(
            loc=self["loc"],
            scale=self["scale"],
            reinterpreted_batch_ndims=self.meta_data["reinterpreted_batch_ndims"],
            shape=self.shape,
            device=self.device,
        )


class TensorTruncatedNormal(TensorDistribution):
    def __init__(
        self,
        loc,
        scale,
        low,
        high,
        reinterpreted_batch_ndims,
        shape=None,
        device=torch.device("cpu"),
    ):
        super().__init__(
            {"loc": loc, "scale": scale},
            shape,
            device,
            {
                "reinterpreted_batch_ndims": reinterpreted_batch_ndims,
                "low": low,
                "high": high,
            },
        )

    def dist(self) -> Distribution:
        return Independent(
            TruncatedNormal(
                self["loc"].float(),
                self["scale"].float(),
                self.meta_data["low"],
                self.meta_data["high"],
            ),
            self.meta_data["reinterpreted_batch_ndims"],
        )

    def copy(self):
        return TensorTruncatedNormal(
            loc=self["loc"],
            scale=self["scale"],
            low=self.meta_data["low"],
            high=self.meta_data["high"],
            reinterpreted_batch_ndims=self.meta_data["reinterpreted_batch_ndims"],
            shape=self.shape,
            device=self.device,
        )


class TensorBernoulli(TensorDistribution):
    def __init__(
        self,
        probs=None,
        logits=None,
        reinterpreted_batch_ndims=0,
        shape=None,
        device=torch.device("cpu"),
    ):
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            data = {"probs": probs}
        else:
            data = {"logits": logits}
        if shape is None:
            if probs is not None:
                shape = probs.shape
            else:
                shape = logits.shape
        super().__init__(
            data,
            shape,
            device,
            {"reinterpreted_batch_ndims": reinterpreted_batch_ndims},
        )

    @property
    def probs(self):
        if "probs" not in self.data:
            self.data["probs"] = torch.sigmoid(self["logits"])
        return self["probs"]

    @property
    def logits(self):
        if "logits" not in self.data:
            self.data["logits"] = torch.log(self["probs"] / (1 - self["probs"]))
        return self["logits"]

    def dist(self):
        if "probs" in self.data:
            return Independent(
                torch.distributions.Bernoulli(
                    probs=self["probs"],
                ),
                self.meta_data["reinterpreted_batch_ndims"],
            )
        else:
            return Independent(
                torch.distributions.Bernoulli(
                    logits=self["logits"],
                ),
                self.meta_data["reinterpreted_batch_ndims"],
            )

    def copy(self):
        if "probs" in self.data:
            return TensorBernoulli(
                probs=self["probs"].clone(),
                reinterpreted_batch_ndims=self.meta_data["reinterpreted_batch_ndims"],
                shape=self.shape,
                device=self.device,
            )
        else:
            return TensorBernoulli(
                logits=self["logits"].clone(),
                reinterpreted_batch_ndims=self.meta_data["reinterpreted_batch_ndims"],
                shape=self.shape,
                device=self.device,
            )


class TensorSoftBernoulli(TensorDistribution):
    def __init__(
        self,
        probs=None,
        logits=None,
        reinterpreted_batch_ndims=0,
        shape=None,
        device=torch.device("cpu"),
    ):
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            data = {"probs": probs}
        else:
            data = {"logits": logits}
        if shape is None:
            if probs is not None:
                shape = probs.shape
            else:
                shape = logits.shape
        super().__init__(
            data,
            shape,
            device,
            {"reinterpreted_batch_ndims": reinterpreted_batch_ndims},
        )

    @property
    def probs(self):
        if "probs" not in self.data:
            self.data["probs"] = torch.sigmoid(self["logits"])
        return self["probs"]

    @property
    def logits(self):
        if "logits" not in self.data:
            self.data["logits"] = torch.log(self["probs"] / (1 - self["probs"]))
        return self["logits"]

    def dist(self):
        if "probs" in self.data:
            return Independent(
                SoftBernoulli(
                    probs=self["probs"],
                ),
                self.meta_data["reinterpreted_batch_ndims"],
            )
        else:
            return Independent(
                SoftBernoulli(
                    logits=self["logits"],
                ),
                self.meta_data["reinterpreted_batch_ndims"],
            )

    def copy(self):
        if "probs" in self.data:
            return TensorSoftBernoulli(
                probs=self["probs"].clone(),
                reinterpreted_batch_ndims=self.meta_data["reinterpreted_batch_ndims"],
                shape=self.shape,
                device=self.device,
            )
        else:
            return TensorSoftBernoulli(
                logits=self["logits"].clone(),
                reinterpreted_batch_ndims=self.meta_data["reinterpreted_batch_ndims"],
                shape=self.shape,
                device=self.device,
            )


class TensorCategorical(TensorDistribution):
    def __init__(
        self,
        logits,
        output_shape,
        shape=None,
        device=torch.device("cpu"),
    ):
        super().__init__(
            {"logits": logits},
            shape,
            device,
            {
                "output_shape": output_shape,
            },
        )

    def dist(self):
        logits = self["logits"].float()
        output_shape = self.meta_data["output_shape"]
        logits = logits.view(*logits.shape[:-1], -1, *output_shape)
        one_hot = OneHotCategoricalStraightThrough(logits=logits)

        return Independent(one_hot, len(output_shape))


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
    def __init__(
        self,
        loc,
        scale,
        reinterpreted_batch_ndims=1,
        shape=None,
        device=torch.device("cpu"),
    ):
        super().__init__(
            {"loc": loc, "scale": scale},
            shape,
            device,
            {"reinterpreted_batch_ndims": reinterpreted_batch_ndims},
        )

    def dist(self) -> Distribution:
        return Independent(
            SamplingDistribution(
                TransformedDistribution(
                    Normal(self["loc"].float(), self["scale"].float()),
                    [
                        ClampedTanhTransform(),
                    ],
                ),
            ),
            self.meta_data["reinterpreted_batch_ndims"],
        )

    def copy(self):
        return TensorTanhNormal(
            loc=self["loc"],
            scale=self["scale"],
            reinterpreted_batch_ndims=self.meta_data["reinterpreted_batch_ndims"],
            shape=self.shape,
            device=self.device,
        )
