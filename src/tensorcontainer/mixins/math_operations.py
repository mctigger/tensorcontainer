"""Mathematical operation mixin for TensorContainer.

This mixin provides mathematical operations that are applied element-wise to all
tensors in the container.
"""

from __future__ import annotations

from typing_extensions import Self

from tensorcontainer.protocols import TensorContainerProtocol


class TensorMathOperationsMixin(TensorContainerProtocol):
    """Mixin providing mathematical operations for tensor containers.

    This mixin contains operations that perform mathematical transformations
    on all tensors within the container. All operations are applied element-wise
    and preserve the container structure.

    Mathematical operations include:
    - Unary operations: abs, sqrt, log, neg
    - Binary operations: add, sub, mul, div, pow
    - Clamping operations: clamp

    Note:
        This mixin should only be used with classes that implement TensorContainerProtocol.
        The protocol constraint is enforced at the class definition level where the mixin
        is used (e.g., TensorDict, TensorDataClass).
    """

    def abs(self) -> Self:
        """Computes the absolute value of each tensor in the container.

        Returns:
            TensorContainer: Container with absolute values of all tensors

        Example:
            >>> container = MyContainer({'a': torch.tensor([-1, 2, -3])})
            >>> abs_container = container.abs()
            >>> # abs_container['a'] == tensor([1, 2, 3])
        """
        return self._tree_map(lambda x: x.abs(), self)

    def add(self, other) -> Self:
        """Adds a value to each tensor in the container.

        Args:
            other: Value to add (scalar, tensor, or compatible container)

        Returns:
            TensorContainer: Container with added values

        Example:
            >>> container = MyContainer({'a': torch.tensor([1, 2, 3])})
            >>> result = container.add(10)
            >>> # result['a'] == tensor([11, 12, 13])
        """
        return self._tree_map(lambda x: x.add(other), self)

    def sub(self, other) -> Self:
        """Subtracts a value from each tensor in the container.

        Args:
            other: Value to subtract (scalar, tensor, or compatible container)

        Returns:
            TensorContainer: Container with subtracted values

        Example:
            >>> container = MyContainer({'a': torch.tensor([10, 20, 30])})
            >>> result = container.sub(5)
            >>> # result['a'] == tensor([5, 15, 25])
        """
        return self._tree_map(lambda x: x.sub(other), self)

    def mul(self, other) -> Self:
        """Multiplies each tensor in the container by a value.

        Args:
            other: Value to multiply by (scalar, tensor, or compatible container)

        Returns:
            TensorContainer: Container with multiplied values

        Example:
            >>> container = MyContainer({'a': torch.tensor([1, 2, 3])})
            >>> result = container.mul(2)
            >>> # result['a'] == tensor([2, 4, 6])
        """
        return self._tree_map(lambda x: x.mul(other), self)

    def div(self, other) -> Self:
        """Divides each tensor in the container by a value.

        Args:
            other: Value to divide by (scalar, tensor, or compatible container)

        Returns:
            TensorContainer: Container with divided values

        Example:
            >>> container = MyContainer({'a': torch.tensor([10, 20, 30])})
            >>> result = container.div(2)
            >>> # result['a'] == tensor([5, 10, 15])
        """
        return self._tree_map(lambda x: x.div(other), self)

    def pow(self, exponent) -> Self:
        """Raises each tensor in the container to a power.

        Args:
            exponent: Exponent value (scalar, tensor, or compatible container)

        Returns:
            TensorContainer: Container with powered values

        Example:
            >>> container = MyContainer({'a': torch.tensor([2, 3, 4])})
            >>> result = container.pow(2)
            >>> # result['a'] == tensor([4, 9, 16])
        """
        return self._tree_map(lambda x: x.pow(exponent), self)

    def sqrt(self) -> Self:
        """Computes the square root of each tensor in the container.

        Returns:
            TensorContainer: Container with square roots of all tensors

        Example:
            >>> container = MyContainer({'a': torch.tensor([4, 9, 16])})
            >>> result = container.sqrt()
            >>> # result['a'] == tensor([2, 3, 4])
        """
        return self._tree_map(lambda x: x.sqrt(), self)

    def log(self) -> Self:
        """Computes the natural logarithm of each tensor in the container.

        Returns:
            TensorContainer: Container with natural logarithms of all tensors

        Example:
            >>> container = MyContainer({'a': torch.tensor([1, 2, 3])})
            >>> result = container.log()
            >>> # result['a'] == tensor([0, 0.693, 1.099])
        """
        return self._tree_map(lambda x: x.log(), self)

    def neg(self) -> Self:
        """Negates each tensor in the container.

        Returns:
            TensorContainer: Container with negated values of all tensors

        Example:
            >>> container = MyContainer({'a': torch.tensor([1, -2, 3])})
            >>> result = container.neg()
            >>> # result['a'] == tensor([-1, 2, -3])
        """
        return self._tree_map(lambda x: x.neg(), self)

    def clamp(self, min, max) -> Self:
        """Clamps each tensor in the container to a range.

        Args:
            min: Minimum value for clamping
            max: Maximum value for clamping

        Returns:
            TensorContainer: Container with clamped values

        Example:
            >>> container = MyContainer({'a': torch.tensor([0, 5, 10])})
            >>> result = container.clamp(2, 8)
            >>> # result['a'] == tensor([2, 5, 8])
        """
        return self._tree_map(lambda x: x.clamp(min, max), self)
