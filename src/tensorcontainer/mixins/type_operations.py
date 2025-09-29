"""Type conversion operation mixin for TensorContainer.

This mixin provides type conversion operations that cast all tensors in the container
to different data types.
"""

from __future__ import annotations

from typing_extensions import Self

from tensorcontainer.protocols import TensorContainerProtocol


class TensorTypeOperationsMixin(TensorContainerProtocol):
    """Mixin providing type conversion operations for tensor containers.

    This mixin contains operations that convert the data type of all tensors
    within the container. Type conversions preserve the container structure
    and tensor shapes while changing the underlying data representation.

    Type conversion operations include:
    - Floating point types: float, double, half
    - Integer types: int, long

    Note:
        This mixin should only be used with classes that implement TensorContainerProtocol.
        The protocol constraint is enforced at the class definition level where the mixin
        is used (e.g., TensorDict, TensorDataClass).
    """

    def float(self) -> Self:
        """Casts all tensors to float type.

        Returns:
            TensorContainer: Container with all tensors cast to torch.float32

        Example:
            >>> container = MyContainer({'a': torch.tensor([1, 2, 3], dtype=torch.int)})
            >>> float_container = container.float()
            >>> # float_container['a'].dtype == torch.float32
        """
        return self._tree_map(lambda x: x.float(), self)

    def double(self) -> Self:
        """Casts all tensors to double type.

        Returns:
            TensorContainer: Container with all tensors cast to torch.float64

        Example:
            >>> container = MyContainer({'a': torch.tensor([1.0, 2.0, 3.0])})
            >>> double_container = container.double()
            >>> # double_container['a'].dtype == torch.float64
        """
        return self._tree_map(lambda x: x.double(), self)

    def half(self) -> Self:
        """Casts all tensors to half type.

        Returns:
            TensorContainer: Container with all tensors cast to torch.float16

        Example:
            >>> container = MyContainer({'a': torch.tensor([1.0, 2.0, 3.0])})
            >>> half_container = container.half()
            >>> # half_container['a'].dtype == torch.float16
        """
        return self._tree_map(lambda x: x.half(), self)

    def long(self) -> Self:
        """Casts all tensors to long type.

        Returns:
            TensorContainer: Container with all tensors cast to torch.int64

        Example:
            >>> container = MyContainer({'a': torch.tensor([1, 2, 3], dtype=torch.int32)})
            >>> long_container = container.long()
            >>> # long_container['a'].dtype == torch.int64
        """
        return self._tree_map(lambda x: x.long(), self)

    def int(self) -> Self:
        """Casts all tensors to int type.

        Returns:
            TensorContainer: Container with all tensors cast to torch.int32

        Example:
            >>> container = MyContainer({'a': torch.tensor([1.5, 2.7, 3.9])})
            >>> int_container = container.int()
            >>> # int_container['a'].dtype == torch.int32
            >>> # int_container['a'] == tensor([1, 2, 3])
        """
        return self._tree_map(lambda x: x.int(), self)
