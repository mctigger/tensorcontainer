"""Device and memory operation mixin for TensorContainer.

This mixin provides operations for moving tensors between devices, creating copies,
and managing memory layout.
"""

from __future__ import annotations

import torch
import torch.utils._pytree as pytree
from typing_extensions import Self

from tensorcontainer.protocols import TensorContainerProtocol


class TensorDeviceOperationsMixin(TensorContainerProtocol):
    """Mixin providing device and memory operations for tensor containers.

    This mixin contains operations that manage tensor device placement, create
    tensor copies, and control memory management. These operations preserve the
    container structure while transforming the underlying tensor storage.

    Device and memory operations include:
    - Device movement: to, cpu, cuda
    - Memory operations: clone, copy, detach

    Note:
        This mixin should only be used with classes that implement TensorContainerProtocol.
        The protocol constraint is enforced at the class definition level where the mixin
        is used (e.g., TensorDict, TensorDataClass).
    """

    def to(self, *args, **kwargs) -> Self:
        """Move and/or cast the container to a device or dtype.

        This method can be used to move tensors to a different device, change their
        dtype, or both. It supports all arguments that torch.Tensor.to() accepts.

        Args:
            *args: Positional arguments passed to torch.Tensor.to()
            **kwargs: Keyword arguments passed to torch.Tensor.to()

        Returns:
            TensorContainer: Container with tensors moved/cast as specified

        Example:
            >>> container = MyContainer({'a': torch.tensor([1, 2, 3])})
            >>> # Move to GPU
            >>> cuda_container = container.to('cuda')
            >>> # Change dtype
            >>> float_container = container.to(torch.float32)
            >>> # Move to GPU and change dtype
            >>> cuda_float = container.to('cuda', torch.float32)
        """
        with self.unsafe_construction():
            leaves, context = self._pytree_flatten()
            leaves = [leaf.to(*args, **kwargs) for leaf in leaves]
            tc = self._pytree_unflatten(leaves, context)

        device = self.device

        is_device_in_args = len(args) > 0 and isinstance(args[0], (str, torch.device))
        is_device_in_kwargs = len(kwargs) > 0 and "device" in kwargs

        if is_device_in_args or is_device_in_kwargs:
            device = pytree.tree_leaves(tc)[0].device

        tc.device = device

        return tc

    def detach(self) -> Self:
        """Returns a new container with all tensors detached from the computation graph.

        Creates a new container where all tensors are detached from the current
        computation graph. The returned tensors will not require gradients.

        Returns:
            TensorContainer: Container with detached tensors

        Example:
            >>> container = MyContainer({'a': torch.tensor([1.0, 2.0], requires_grad=True)})
            >>> detached = container.detach()
            >>> # detached['a'].requires_grad == False
        """
        return self._tree_map(lambda x: x.detach(), self)

    def clone(self, *, memory_format: torch.memory_format | None = None) -> Self:
        """Create a deep copy of the container with optional memory format control.

        Creates a new container with cloned tensors. All tensor data is copied,
        but metadata (shape, device) is shallow-copied. Supports memory format
        specification for performance optimization.

        Args:
            memory_format: Memory layout for cloned tensors. Defaults to preserve_format.
                          Options: torch.contiguous_format, torch.channels_last, etc.

        Returns:
            TensorContainer: Deep copy of the container

        Example:
            >>> cloned = container.clone()  # Deep copy with preserved layout
            >>>
            >>> # Force contiguous memory layout for performance
            >>> contiguous = container.clone(memory_format=torch.contiguous_format)
            >>>
            >>> # Clone preserves independence
            >>> cloned[0] = new_data  # Original container unchanged
        """
        cloned_td = self._tree_map(lambda x: x.clone(memory_format=memory_format), self)
        return cloned_td

    def copy(self) -> Self:
        """Create a shallow copy of the container.

        Creates a new container with the same tensor references. This is a
        lightweight operation that copies the container structure but not
        the tensor data.

        Returns:
            TensorContainer: Shallow copy of the container

        Example:
            >>> copied = container.copy()
            >>> # copied and container share the same tensor data
            >>> # but have independent container structures
        """
        return self._tree_map(lambda x: x, self)

    def cpu(self) -> Self:
        """Returns a new container with all tensors on the CPU.

        Moves all tensors in the container to CPU memory.

        Returns:
            TensorContainer: Container with all tensors on CPU

        Example:
            >>> gpu_container = MyContainer({'a': torch.tensor([1, 2, 3]).cuda()})
            >>> cpu_container = gpu_container.cpu()
            >>> # cpu_container['a'].device.type == 'cpu'
        """
        return self.to("cpu")

    def cuda(self, device=None, non_blocking: bool = False) -> Self:
        """Returns a new container with all tensors on the specified CUDA device.

        Args:
            device: CUDA device index. If None, uses the current CUDA device.
            non_blocking: If True, tries to convert asynchronously when possible.

        Returns:
            TensorContainer: Container with all tensors on CUDA device

        Example:
            >>> cpu_container = MyContainer({'a': torch.tensor([1, 2, 3])})
            >>> gpu_container = cpu_container.cuda()
            >>> # gpu_container['a'].device.type == 'cuda'
            >>>
            >>> # Specific device
            >>> gpu1_container = cpu_container.cuda(1)
            >>> # gpu1_container['a'].device == torch.device('cuda:1')
        """
        return self.to(
            f"cuda:{device}" if device is not None else "cuda",
            non_blocking=non_blocking,
        )
