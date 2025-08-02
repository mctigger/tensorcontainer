from __future__ import annotations

from torch import Tensor
from torch.distributions.utils import broadcast_all as torch_broadcast_all
from torch.types import Number


def broadcast_all(*values: Number | Tensor) -> tuple[Tensor, ...]:
    """
    Broadcast all input values to a common shape.

    Given a list of values (possibly containing numbers), returns a tuple where each
    value is broadcasted based on the following rules:
      - `torch.Tensor` instances are broadcasted as per broadcasting semantics.
      - Number instances (scalars) are upcast to tensors having
        the same size and type as the first tensor passed to `values`.  If all the
        values are scalars, then they are upcasted to scalar Tensors.

    Args:
        *values: Variable number of arguments, each being a Number | Tensor,
                or objects implementing __torch_function__

    Returns:
        tuple[Tensor, ...]: Tuple of broadcasted tensors

    Raises:
        ValueError: if any of the values is not a Number instance,
            a torch.Tensor instance, or an instance implementing __torch_function__

    Example:
        >>> import torch
        >>> from tensorcontainer.tensor_distribution.utils import broadcast_all
        >>> loc = torch.tensor([0.0, 1.0])
        >>> scale = 1.0
        >>> broadcasted_loc, broadcasted_scale = broadcast_all(loc, scale)
        >>> broadcasted_loc.shape
        torch.Size([2])
        >>> broadcasted_scale.shape
        torch.Size([2])
    """
    return torch_broadcast_all(*values)
