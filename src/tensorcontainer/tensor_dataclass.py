from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass, fields
from typing import Optional, Union
from typing_extensions import Self
import torch
from torch import Tensor
from tensorcontainer.types import DeviceLike, ShapeLike
from typing_extensions import dataclass_transform

from tensorcontainer.tensor_annotated import TensorAnnotated
from tensorcontainer.tensor_container import TensorContainer
from tensorcontainer.mixins import (
    TensorShapeOperationsMixin,
    TensorMathOperationsMixin,
    TensorTypeOperationsMixin,
    TensorDeviceOperationsMixin,
)

TDCompatible = Union[Tensor, TensorContainer]


@dataclass_transform(eq_default=False)
class TensorDataclassTransform:
    """This class is just needed for type hints. Directly decorating TensorDataclass with @dataclass_transform(eq_default=False) does not work."""

    pass


class TensorDataClass(
    TensorAnnotated,
    TensorShapeOperationsMixin,
    TensorMathOperationsMixin,
    TensorTypeOperationsMixin,
    TensorDeviceOperationsMixin,
    TensorDataclassTransform,
):
    """A dataclass TensorContainer.

    A class that inherits from TensorDataClass will automatically be converted to a
    @dataclass. Any annotated field whos value is a Tensor or TensorContainer will
    transform according to batch/event semantics. Every annotated field whos value is
    of any other type will be regarded as meta data.

    Inherits batch/event dimension semantics from TensorContainer and annotation-based
    field separation from TensorAnnotated. See those classes for core tensor container
    concepts and PyTree integration details.

    ## Automatic Dataclass Conversion

    Any subclass is automatically converted to a dataclass:

    ```python
    class MyData(TensorDataClass):
        features: torch.Tensor
        labels: torch.Tensor
        some_metadata: str = "default"

    my_data = MyData(features=torch.rand(2, 3, 4), labels=torch.rand(2, 3, 4), some_metadata="custom")
    ```

    ## Field Inheritance and Composition

    ```python
    class BaseData(TensorDataClass):
        observations: torch.Tensor

    class ExtendedData(BaseData):
        actions: torch.Tensor      # Inherits observations
        rewards: torch.Tensor
    ```

    ## TensorDataClass vs TensorDict

    | Feature | TensorDataClass | TensorDict |
    |---------|-----------------|------------|
    | Access Pattern | `obj.field` | `obj["key"]` |
    | Type Safety | Static typing | Runtime checks |
    | IDE Support | Full autocomplete | Limited |
    | Field Definition | Compile-time | Runtime |
    | Dynamic Fields | Not supported | Full support |

    Args:
        shape: Batch shape for tensor field validation (see TensorContainer).
        device: Target device for tensors (see TensorContainer).

    Raises:
        TypeError: If eq=True is specified (incompatible with tensor fields).

    Note:
        Device and shape validation behavior inherited from TensorContainer.
        Field separation behavior inherited from TensorAnnotated.
    """

    # The only reason we define shape and device here is such that @dataclass_transform
    # can enable static analyzers to provide type hints in IDEs. Both are programmatically
    # added in __init_subclass__ so removing the following two lines will only remove the
    # type hints, but the class will stay functional.
    shape: ShapeLike
    device: DeviceLike

    def __init_subclass__(cls, **kwargs):
        """Automatically convert subclasses to dataclasses with proper field inheritance.

        We want to enforce that shape and device must be in __init__ and are available as
        attributes.

        Args:
            **kwargs: Class definition arguments, may include dataclass options.

        Raises:
            TypeError: If eq=True is specified (incompatible with tensor fields).
        """
        # This check is needed as slots=True will result in dataclass(cls) creating a new class
        # and thus triggering __init__subclass again. However, we already have ran __init__subclass__
        # already for this class. To avoid infinte recursion, we have the following check.
        # Note: We check that __slots__ is non-empty to avoid early return when __slots__ = ()
        # is inherited from typing.Protocol in the class hierarchy.
        if hasattr(cls, "__slots__") and cls.__slots__:
            return

        annotations = cls._get_annotations(TensorDataClass)

        cls.__annotations__ = {
            "shape": ShapeLike,
            "device": Optional[torch.device],
            **annotations,
        }

        # Get valid dataclass parameters dynamically
        dataclass_params = set(inspect.signature(dataclass).parameters.keys()) - {"cls"}
        dc_kwargs = {
            k: kwargs.pop(k) for k in list(kwargs.keys()) if k in dataclass_params
        }

        super().__init_subclass__(**kwargs)

        # There is no way to say whether to TensorDataClasses are equal or not at the moment.
        # Something like torch.eq on every tensor would be thinkable.
        if dc_kwargs.get("eq") is True:
            raise TypeError(
                f"Cannot create {cls.__name__} with eq=True. TensorDataClass requires eq=False."
            )
        dc_kwargs.setdefault("eq", False)

        dataclass(cls, **dc_kwargs)

    def __post_init__(self):
        """After dataclasses.dataclass __init__ is called, we use this method to
        pass the shape and device to the parent class.

        Raises:
            RuntimeError: If validation fails (see TensorContainer._validate for details).
        """
        super().__init__(self.shape, self.device)

    def __copy__(self: Self) -> Self:
        """Create torch.compile-safe shallow copy with shared tensor data.

        Solves the problem of copy.copy() causing graph breaks in torch.compile.
        Manually copies field references while sharing underlying tensor data.

        Returns:
            Self: New instance with shared tensor data.

        Note:
            For independent tensor data, use clone() inherited from TensorContainer.
        """
        # Create a new, uninitialized instance of the correct class.
        cls = type(self)
        new_obj = cls.__new__(cls)

        # Manually copy all dataclass fields.
        for field in fields(self):
            value = getattr(self, field.name)
            setattr(new_obj, field.name, value)

        # Manually call __post_init__ to initialize the TensorContainer part
        # and run validation logic. This is necessary because we bypassed __init__.
        if hasattr(new_obj, "__post_init__"):
            new_obj.__post_init__()

        return new_obj

    def __deepcopy__(self: Self, memo: dict | None = None) -> Self:
        """Create torch.compile-safe deep copy with cloned tensor data.

        Solves the problem of copy.deepcopy() causing graph breaks in torch.compile.
        Manually handles field copying with circular reference protection.

        Args:
            memo: Tracks copied objects to prevent infinite recursion.

        Returns:
            Self: New instance with cloned tensor data and copied metadata.
        """
        if memo is None:
            memo = {}

        cls = type(self)
        # Check if the object is already in memo
        if id(self) in memo:
            return memo[id(self)]

        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj

        for field in fields(self):
            value = getattr(self, field.name)
            # The `shape` and `device` fields are part of the dataclass fields
            # due to their annotations in TensorDataclass.
            # These should be deepcopied as well if they are not None.
            if field.name in ("shape", "device"):
                # Tuples (shape) and torch.device are immutable or behave as such.
                # Direct assignment is fine and avoids torch.compile issues with deepcopying them.
                # Direct assignment for immutable types like tuple (shape) and torch.device.
                # This avoids torch.compile issues with copy.copy or copy.deepcopy on these types.
                setattr(new_obj, field.name, value)
            elif isinstance(value, Tensor):
                # For torch.Tensor, use .clone() for a deep copy of data.
                setattr(new_obj, field.name, value.clone())
            elif isinstance(value, list):
                # For lists, create a new list. This is a shallow copy of the list structure.
                # If list items are mutable and need deepcopying, torch.compile might
                # still struggle with a generic deepcopy of those items.
                # For a list of immutables (like in the test), this is effectively a deepcopy.
                setattr(new_obj, field.name, list(value))
            else:
                # For other fields (e.g., dict, other custom objects), attempt deepcopy.
                # This remains a potential point of failure for torch.compile
                # if it doesn't support deepcopying these specific types.
                setattr(new_obj, field.name, copy.deepcopy(value))

        # Manually call __post_init__ to initialize the TensorContainer part
        # and run validation logic. This is necessary because we bypassed __init__.
        # __post_init__ in TensorDataclass handles shape and device initialization
        # and validation, which is crucial after all fields are set.
        if hasattr(new_obj, "__post_init__"):
            new_obj.__post_init__()

        return new_obj
