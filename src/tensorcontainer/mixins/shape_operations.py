"""Shape operation mixin for TensorContainer.

This mixin provides shape transformation operations that work on the batch dimensions
of tensor containers while preserving event dimensions.
"""

from __future__ import annotations

import torch
from typing_extensions import Self

from tensorcontainer.protocols import TensorContainerProtocol


class TensorShapeOperationsMixin(TensorContainerProtocol):
    """Mixin providing shape transformation operations for tensor containers.

    This mixin contains operations that manipulate the batch dimensions of the container
    while automatically preserving event dimensions of individual tensors.

    Shape operations include:
    - Dimension transformations: view, reshape, expand
    - Dimension reordering: permute, transpose, t
    - Dimension manipulation: squeeze, unsqueeze
    - Dimension queries: size, dim, numel

    Note:
        This mixin should only be used with classes that implement TensorContainerProtocol.
        The protocol constraint is enforced at the class definition level where the mixin
        is used (e.g., TensorDict, TensorDataClass).
    """

    def view(self, *shape: int) -> Self:
        """Return a view with modified batch dimensions, preserving event dimensions.

        Creates a view of the container with new batch shape while preserving all
        event dimensions. The total number of elements in batch dimensions must remain
        the same (view constraint).

        Args:
            *shape: New batch shape dimensions

        Returns:
            TensorContainer: View with new batch shape

        Example:
            >>> container.shape == (4, 3)  # 12 batch elements
            >>> # Reshape batch dimensions while preserving event dims
            >>> viewed = container.view(2, 6)    # batch becomes (2, 6)
            >>> viewed = container.view(12)      # batch becomes (12,)
            >>> viewed = container.view(-1, 3)   # batch becomes (4, 3) - inferred
            >>>
            >>> # If tensors have event dims, they are preserved:
            >>> # Original: tensor.shape == (4, 3, 128)  # event dims (128,)
            >>> # After view(2, 6): tensor.shape == (2, 6, 128)
        """
        return self._tree_map(lambda x: x.view(*shape, *x.shape[self.ndim :]), self)

    def reshape(self, *shape: int) -> Self:
        """Return a reshaped container with modified batch dimensions.

        Reshapes the batch dimensions while preserving event dimensions. Unlike view(),
        reshape() can change the memory layout if needed and doesn't require the
        tensor to be contiguous.

        Args:
            *shape: New batch shape dimensions

        Returns:
            TensorContainer: Reshaped container

        Example:
            >>> container.shape == (4, 3)  # 12 batch elements
            >>> reshaped = container.reshape(2, 6)   # batch becomes (2, 6)
            >>> reshaped = container.reshape(-1)     # batch becomes (12,)
            >>>
            >>> # Handles non-contiguous tensors unlike view()
            >>> transposed = container.transpose(0, 1)  # Non-contiguous
            >>> reshaped = transposed.reshape(6, 2)     # Works (reshape can copy)
        """
        return self._tree_map(lambda x: x.reshape(*shape, *x.shape[self.ndim :]), self)

    def expand(self, *shape: int) -> Self:
        """Expand the container to a larger batch size.

        Returns a new view of the container tensor with singleton dimensions expanded
        to a larger size. Only dimensions of size 1 can be expanded.

        Args:
            *shape: The desired expanded batch shape

        Returns:
            TensorContainer: Container with expanded batch dimensions
        """
        return self._tree_map(lambda x: x.expand(*shape, *x.shape[self.ndim :]), self)

    def permute(self, *dims: int) -> Self:
        """Permutes the batch dimensions of the container.

        This is equivalent to calling :meth:`torch.Tensor.permute` on each tensor
        in the container, but only for the batch dimensions.

        Args:
            *dims (int): The desired ordering of dimensions.

        Returns:
            A new container with the batch dimensions permuted.
        """
        if len(dims) != self.ndim:
            raise RuntimeError(
                f"permute() expected {self.ndim} dimensions but got {len(dims)}"
            )
        if len(set(dims)) != len(dims):
            raise RuntimeError("permute(): duplicate dimensions are not allowed")
        for dim in dims:
            if not 0 <= dim < self.ndim:
                raise RuntimeError(
                    f"permute(): dimension out of range (expected to be in range of [0, {self.ndim - 1}], but got {dim})"
                )
        return self._tree_map(
            lambda x: x.permute(*dims, *range(self.ndim, x.ndim)), self
        )

    def squeeze(self, dim: int | None = None) -> Self:
        """Squeezes the batch dimensions of the container.

        Args:
            dim (int, optional): The dimension to squeeze. If ``None``, all
                batch dimensions of size 1 are squeezed.

        Returns:
            A new container with the specified dimensions squeezed.
        """
        if dim is not None:
            if self.shape[dim] != 1:
                return self.clone()
            new_shape = list(self.shape)
            new_shape.pop(dim)
            return self.reshape(*new_shape)
        else:
            new_shape = [s for s in self.shape if s != 1]
            if len(new_shape) == len(self.shape):
                return self.clone()
            return self.reshape(*new_shape)

    def t(self) -> Self:
        """Transposes the first two batch dimensions of the container.

        This is equivalent to ``self.transpose(0, 1)``.

        Returns:
            A new container with the first two batch dimensions transposed.
        """
        if self.ndim < 2:
            raise RuntimeError(
                "t() expects a tensor with at least 2 dimensions, but got a tensor with "
                f"{self.ndim} dimensions instead"
            )
        return self.transpose(0, 1)

    def transpose(self, dim0: int, dim1: int) -> Self:
        """Transposes two batch dimensions of the container.

        Args:
            dim0 (int): The first dimension to transpose.
            dim1 (int): The second dimension to transpose.

        Returns:
            A new container with the specified dimensions transposed.
        """
        return self._tree_map(lambda x: x.transpose(dim0, dim1), self)

    def unsqueeze(self, dim: int) -> Self:
        """Unsqueezes a batch dimension of the container.

        Args:
            dim (int): The dimension to unsqueeze.

        Returns:
            A new container with the specified dimension unsqueezed.
        """
        new_shape = torch.empty(self.shape).unsqueeze(dim).shape
        return self.reshape(*new_shape)

    def size(self) -> torch.Size:
        """Returns the size of the batch dimensions."""
        return self.shape

    def dim(self) -> int:
        """Returns the number of batch dimensions."""
        return self.ndim

    def numel(self) -> int:
        """Returns the total number of elements in the batch dimensions."""
        return self.size().numel()
