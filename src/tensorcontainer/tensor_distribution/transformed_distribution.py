from __future__ import annotations

from typing import Any

from torch.distributions import Distribution
from torch.distributions import TransformedDistribution as TorchTransformedDistribution
from torch.distributions.transforms import Transform

from tensorcontainer.tensor_distribution.base import TensorDistribution


class TransformedDistribution(TensorDistribution):
    """
    Creates a transformed distribution.

    Args:
        base_distribution (TensorDistribution): The base distribution.
        transforms (list[Transform]): A list of transforms.
    """

    base_distribution: TensorDistribution
    transforms: list[Transform]

    def __init__(
        self,
        base_distribution: TensorDistribution,
        transforms: list[Transform],
        validate_args: bool | None = None,
    ) -> None:
        self.base_distribution = base_distribution
        self.transforms = transforms
        super().__init__(
            base_distribution.batch_shape, base_distribution.device, validate_args
        )

    @classmethod
    def _unflatten_distribution(
        cls, attributes: dict[str, Any]
    ) -> TransformedDistribution:
        return cls(
            base_distribution=attributes["base_distribution"],
            transforms=attributes["transforms"],
            validate_args=attributes.get("_validate_args"),
        )

    def dist(self) -> Distribution:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return TorchTransformedDistribution(
            base_distribution=self.base_distribution.dist(),
            transforms=self.transforms,
            validate_args=self._validate_args,
        )
