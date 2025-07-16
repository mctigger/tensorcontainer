import inspect

import torch
from typing import Optional
from torch import Tensor

from tensorcontainer.tensor_annotated import TensorAnnotated
import pytest


@pytest.fixture
def single_parent_class():
    class A(TensorAnnotated):
        x: Tensor
        def __init__(self, x: Tensor):
            self.x = x
            super().__init__(shape=x.shape, device=x.device)

    class B(A):
        y: Tensor
        def __init__(self, x: Tensor, y: Tensor, **kwargs): # **kwargs captures device and shape in __init_from_reconstructed
            self.y = y
            super().__init__(x=x)

    return B


@pytest.fixture
def multiple_tensor_annotated_parents():
    class A(TensorAnnotated):
        x: Tensor
        def __init__(self, *, x: torch.Tensor, **kwargs):
            self.x = x
            super().__init__(**kwargs)


    class B(TensorAnnotated):
        y: Tensor
        def __init__(self, *, y: torch.Tensor, **kwargs):
            self.y = y
            super().__init__(**kwargs)


    class C(B, A):
        z: Tensor
        def __init__(self, x, y, z, **kwargs): # **kwargs captures device and shape in __init_from_reconstructed
            self.z = z
            super().__init__(shape=x.shape, device=x.device, x=x, y=y)
    return C


@pytest.fixture
def single_tensor_annotated_parent():
    class A(TensorAnnotated):
        x: Tensor
        def __init__(self, x: Tensor):
            self.x = x
            super().__init__(shape=x.shape, device=x.device)

    class B:
        pass

    class C(A, B):
        z: Tensor
        def __init__(self, x: torch.Tensor, z: Tensor, **kwargs): # **kwargs captures device and shape in __init_from_reconstructed
            self.z = z
            super().__init__(x=x)
    return C


class TestSubclassing:
    def test_single_parent_class_instantiation(self, single_parent_class):
        x = torch.randn(2, 3,4)
        y = torch.randn(2, 3,4)
        instance = single_parent_class(
            x=x,
            y=y
        )
        instance = instance.copy() # To trigger pytree operations
        assert instance.x  is x
        assert instance.y is y

    def test_multiple_parent_classes_multiple_tensor_annotated_instantiation(
        self, multiple_tensor_annotated_parents
    ):
        x = torch.randn(2, 3,4)
        y = torch.randn(2, 3,4)
        z = torch.randn(2, 3,4)
        instance = multiple_tensor_annotated_parents(
            x=x,
            y=y,
            z=z
        )
        instance = instance.copy() # To trigger pytree operations
        assert instance.x is x
        assert instance.y is y
        assert instance.z is z

    def test_multiple_parent_classes_single_tensor_annotated_instantiation(
        self, single_tensor_annotated_parent
    ):
        x = torch.randn(2, 3,4)
        z =torch.randn(2, 3,4)
        instance = single_tensor_annotated_parent(
            x=x,
            z=z,
        )
        instance = instance.copy() # To trigger pytree operations
        assert instance.x is x
        assert instance.z is z