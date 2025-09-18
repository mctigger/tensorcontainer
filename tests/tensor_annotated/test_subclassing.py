"""Tests for TensorAnnotated subclassing behavior.

This module comprehensively tests inheritance patterns for TensorAnnotated,
ensuring proper PyTree compatibility and attribute handling across various
inheritance scenarios including single inheritance, multiple inheritance,
and mixed inheritance with non-TensorAnnotated classes.
"""
from typing import Optional, Tuple

import pytest
import torch
from torch import Tensor
import torch.utils._pytree as pytree

from tensorcontainer.tensor_annotated import TensorAnnotated


# Test Fixtures and Utilities

@pytest.fixture
def tensor_data():
    """Provides consistent tensor data for tests."""
    shape = (2, 3, 4)
    return {
        'x': torch.randn(shape),
        'y': torch.randn(shape),
        'z': torch.randn(shape),
        'shape': shape,
        'device': torch.device('cpu')
    }


@pytest.fixture
def cross_device_tensor_data():
    """Provides tensor data with different devices for cross-device tests."""
    shape = (2, 3, 4)
    cuda_available = torch.cuda.is_available()

    data = {
        'x_cpu': torch.randn(shape, device='cpu'),
        'y_cpu': torch.randn(shape, device='cpu'),
        'shape': shape,
        'cpu_device': torch.device('cpu')
    }

    if cuda_available:
        data.update({
            'x_cuda': torch.randn(shape, device='cuda'),
            'y_cuda': torch.randn(shape, device='cuda'),
            'cuda_device': torch.device('cuda')
        })

    return data


@pytest.fixture
def multi_shape_tensor_data():
    """Provides tensors with different shapes for batch dimension tests."""
    return {
        'small': torch.randn(2, 3),
        'medium': torch.randn(2, 3, 4),
        'large': torch.randn(2, 3, 4, 5),
        'shapes': [(2, 3), (2, 3, 4), (2, 3, 4, 5)]
    }


def pytree_roundtrip(instance):
    """Test pytree operations by doing a flatten/unflatten roundtrip."""
    leaves, context = pytree.tree_flatten(instance)
    return pytree.tree_unflatten(leaves, context)


def assert_tensor_equal_and_properties(actual: Tensor, expected: Tensor, check_identity: bool = False):
    """Assert tensor equality with comprehensive property validation.

    Args:
        actual: The tensor to validate
        expected: The expected tensor
        check_identity: If True, check tensor identity (same object)
    """
    if check_identity:
        assert actual is expected, "Expected tensor identity to be preserved"
    else:
        assert torch.equal(actual, expected), "Tensor values should be equal"

    assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} != {expected.shape}"
    assert actual.device == expected.device, f"Device mismatch: {actual.device} != {expected.device}"
    assert actual.dtype == expected.dtype, f"Dtype mismatch: {actual.dtype} != {expected.dtype}"


def assert_attribute_preserved(instance, restored_instance, attr_name: str):
    """Assert that an attribute is properly preserved through pytree operations."""
    original_tensor = getattr(instance, attr_name)
    restored_tensor = getattr(restored_instance, attr_name)
    assert_tensor_equal_and_properties(restored_tensor, original_tensor, check_identity=True)


def assert_attribute_missing(instance, attr_name: str):
    """Assert that an attribute is missing from an instance."""
    with pytest.raises(AttributeError, match=f"object has no attribute '{attr_name}'"):
        getattr(instance, attr_name)


# Base Test Classes as Fixtures

class BaseTensorAnnotatedA(TensorAnnotated):
    """Base TensorAnnotated class with single tensor attribute."""
    x: Tensor

    def __init__(self, x: Tensor, **kwargs):
        self.x = x
        init_shape = kwargs.pop("shape", x.shape)
        init_device = kwargs.pop("device", x.device)
        super().__init__(shape=init_shape, device=init_device, **kwargs)


class BaseTensorAnnotatedB(TensorAnnotated):
    """Base TensorAnnotated class with single tensor attribute (y)."""
    y: Tensor

    def __init__(self, *, y: torch.Tensor, **kwargs):
        self.y = y
        super().__init__(**kwargs)


class BaseRegularClass:
    """Regular non-TensorAnnotated class for mixed inheritance tests."""
    def __init__(self, attr_value: Optional[Tensor] = None):
        if attr_value is not None:
            self.non_annotated_attr = attr_value


class TestSubclassingPositive:
    """Tests for positive scenarios in TensorAnnotated subclassing.

    These tests ensure that subclassing TensorAnnotated works as expected
    when attributes are correctly annotated and inherited. All positive
    scenarios should preserve tensor identity through pytree operations.
    """

    def test_single_parent_inheritance(self, tensor_data):
        """Tests simple inheritance from a single TensorAnnotated parent.

        Validates that:
        - Child class properly inherits parent attributes
        - Both parent and child attributes are preserved in pytree operations
        - Tensor identity is maintained through serialization
        """
        class SingleParent(BaseTensorAnnotatedA):
            pass

        class ChildWithAttribute(SingleParent):
            y: Tensor

            def __init__(self, x: Tensor, y: Tensor, **kwargs):
                self.y = y
                super().__init__(x=x, **kwargs)

        instance = ChildWithAttribute(x=tensor_data['x'], y=tensor_data['y'])
        instance_restored = pytree_roundtrip(instance)

        # Verify both attributes are preserved with identity
        assert_attribute_preserved(instance, instance_restored, 'x')
        assert_attribute_preserved(instance, instance_restored, 'y')

    def test_multiple_tensor_annotated_inheritance(self, tensor_data):
        """Tests multiple inheritance with multiple TensorAnnotated parents.

        Validates that:
        - Multiple TensorAnnotated parents are properly handled
        - All annotated attributes from all parents are preserved
        - Method resolution order works correctly
        """
        class MultipleParentChild(BaseTensorAnnotatedB, BaseTensorAnnotatedA):
            z: Tensor

            def __init__(self, x, y, z, **kwargs):
                self.z = z
                # Extract shape and device to avoid conflicts
                init_shape = kwargs.pop("shape", x.shape)
                init_device = kwargs.pop("device", x.device)
                # Initialize both parents with proper MRO
                super().__init__(shape=init_shape, device=init_device, x=x, y=y, **kwargs)

        instance = MultipleParentChild(
            x=tensor_data['x'],
            y=tensor_data['y'],
            z=tensor_data['z']
        )
        instance_restored = pytree_roundtrip(instance)

        # Verify all three attributes are preserved
        assert_attribute_preserved(instance, instance_restored, 'x')
        assert_attribute_preserved(instance, instance_restored, 'y')
        assert_attribute_preserved(instance, instance_restored, 'z')

    def test_mixed_inheritance_single_tensor_annotated(self, tensor_data):
        """Tests mixed inheritance with one TensorAnnotated and regular classes.

        Validates that:
        - Mixed inheritance works correctly
        - Only TensorAnnotated attributes are preserved in pytree operations
        - Regular class inheritance doesn't interfere with tensor handling
        """
        class MixedInheritanceChild(BaseTensorAnnotatedA, BaseRegularClass):
            z: Tensor

            def __init__(self, x: torch.Tensor, z: Tensor, **kwargs):
                self.z = z
                BaseTensorAnnotatedA.__init__(self, x=x, **kwargs)
                BaseRegularClass.__init__(self)

        instance = MixedInheritanceChild(x=tensor_data['x'], z=tensor_data['z'])
        instance_restored = pytree_roundtrip(instance)

        # Verify TensorAnnotated attributes are preserved
        assert_attribute_preserved(instance, instance_restored, 'x')
        assert_attribute_preserved(instance, instance_restored, 'z')

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cross_device_inheritance(self, cross_device_tensor_data):
        """Tests inheritance behavior with tensors on different devices.

        Validates that:
        - Cross-device tensor handling works in inheritance
        - Device information is properly preserved
        - PyTree operations work across devices
        """
        class CrossDeviceChild(BaseTensorAnnotatedA):
            y: Tensor

            def __init__(self, x: Tensor, y: Tensor, **kwargs):
                self.y = y
                # Extract and override device to allow mixed devices
                kwargs.pop("device", None)  # Remove any existing device parameter
                super().__init__(x=x, device=None, **kwargs)

        instance = CrossDeviceChild(
            x=cross_device_tensor_data['x_cpu'],
            y=cross_device_tensor_data['x_cuda']
        )
        instance_restored = pytree_roundtrip(instance)

        # Verify cross-device attributes are preserved
        assert_attribute_preserved(instance, instance_restored, 'x')
        assert_attribute_preserved(instance, instance_restored, 'y')
        assert instance_restored.x.device.type == 'cpu'
        assert instance_restored.y.device.type == 'cuda'

    def test_method_inheritance(self, tensor_data):
        """Tests that methods are properly inherited and work with tensor operations.

        Validates that:
        - Custom methods are inherited correctly
        - Methods can operate on inherited tensor attributes
        - Method behavior is consistent after pytree operations
        """
        class ParentWithMethods(BaseTensorAnnotatedA):
            def get_tensor_sum(self) -> Tensor:
                """Custom method operating on tensor attribute."""
                return self.x.sum()

            def tensor_operation(self) -> Tensor:
                """Method that performs operations on the tensor."""
                return self.x * 2

        class ChildWithMethods(ParentWithMethods):
            y: Tensor

            def __init__(self, x: Tensor, y: Tensor, **kwargs):
                self.y = y
                super().__init__(x=x, **kwargs)

            def combined_operation(self) -> Tensor:
                """Method using both parent and child attributes."""
                return self.x + self.y

        instance = ChildWithMethods(x=tensor_data['x'], y=tensor_data['y'])
        instance_restored = pytree_roundtrip(instance)

        # Verify method inheritance works
        assert hasattr(instance_restored, 'get_tensor_sum')
        assert hasattr(instance_restored, 'tensor_operation')
        assert hasattr(instance_restored, 'combined_operation')

        # Verify methods produce consistent results
        original_sum = instance.get_tensor_sum()
        restored_sum = instance_restored.get_tensor_sum()
        assert torch.equal(original_sum, restored_sum)

        original_combined = instance.combined_operation()
        restored_combined = instance_restored.combined_operation()
        assert torch.equal(original_combined, restored_combined)


class TestSubclassingNegative:
    """Tests for negative scenarios in TensorAnnotated subclassing.

    These tests ensure that attributes are correctly ignored or handled
    when they do not conform to the TensorAnnotated structure, such as
    attributes from non-TensorAnnotated parent classes or non-annotated attributes.

    Key behaviors tested:
    - Non-annotated attributes are excluded from pytree operations
    - Attributes from non-TensorAnnotated parents are ignored
    - Only properly annotated tensor attributes are preserved
    """

    def test_non_tensor_annotated_attributes_excluded(self, tensor_data):
        """Tests that attributes from non-TensorAnnotated parents are excluded.

        Validates that:
        - Attributes from regular (non-TensorAnnotated) parent classes are ignored
        - Only properly annotated tensor attributes are included in pytree operations
        - Mixed inheritance correctly identifies which attributes to preserve
        """
        class RegularParentWithAttribute:
            def __init__(self, y: Optional[Tensor] = None):
                if y is not None:
                    self.y = y  # This should be excluded from pytree

        class MixedInheritanceChild(BaseTensorAnnotatedA, RegularParentWithAttribute):
            z: Tensor  # This should be included

            def __init__(self, x: Tensor, z: Tensor, y: Optional[Tensor] = None, **kwargs):
                self.z = z
                BaseTensorAnnotatedA.__init__(self, x=x, **kwargs)
                RegularParentWithAttribute.__init__(self, y=y)

        instance = MixedInheritanceChild(
            x=tensor_data['x'],
            z=tensor_data['z'],
            y=tensor_data['y']
        )

        # Verify the original instance has all attributes
        assert hasattr(instance, 'x')
        assert hasattr(instance, 'y')
        assert hasattr(instance, 'z')

        instance_restored = pytree_roundtrip(instance)

        # Only TensorAnnotated attributes should be preserved
        assert_attribute_preserved(instance, instance_restored, 'x')
        assert_attribute_preserved(instance, instance_restored, 'z')

        # Non-TensorAnnotated attribute should be missing
        assert_attribute_missing(instance_restored, 'y')

    def test_non_annotated_attributes_excluded(self, tensor_data):
        """Tests that non-annotated attributes are excluded from pytree operations.

        Validates that:
        - Attributes without type annotations are ignored
        - Only properly annotated Tensor attributes are preserved
        - This applies even within TensorAnnotated subclasses
        """
        class ChildWithNonAnnotatedAttribute(BaseTensorAnnotatedA):
            # Note: no type annotation for 'y'
            def __init__(self, x: Tensor, y: Optional[Tensor] = None, **kwargs):
                if y is not None:
                    self.y = y  # Non-annotated attribute - should be excluded
                super().__init__(x=x, **kwargs)

        instance = ChildWithNonAnnotatedAttribute(
            x=tensor_data['x'],
            y=tensor_data['y']
        )

        # Verify original instance has both attributes
        assert hasattr(instance, 'x')
        assert hasattr(instance, 'y')

        instance_restored = pytree_roundtrip(instance)

        # Only annotated attribute should be preserved
        assert_attribute_preserved(instance, instance_restored, 'x')

        # Non-annotated attribute should be missing
        assert_attribute_missing(instance_restored, 'y')

    def test_tensor_annotated_position_in_mro(self, tensor_data):
        """Tests that TensorAnnotated position in MRO affects attribute handling.

        Validates that:
        - Only attributes from classes that properly inherit TensorAnnotated are preserved
        - Attributes from regular classes are excluded regardless of MRO position
        - TensorAnnotated behavior is only applied to direct inheritance chains
        """
        class RegularClassA:
            def __init__(self, x: Tensor):
                self.x = x  # Should be excluded (from regular class)

        class RegularClassB:
            def __init__(self, y: Tensor):
                self.y = y  # Should be excluded (from regular class)

        class MultipleInheritanceWithTensorAnnotated(RegularClassA, RegularClassB, TensorAnnotated):
            z: Tensor  # Should be included (annotated in TensorAnnotated subclass)

            def __init__(
                self,
                z: Tensor,
                x: Optional[Tensor] = None,
                y: Optional[Tensor] = None,
                **kwargs,
            ):
                if x is not None:
                    RegularClassA.__init__(self, x=x)
                if y is not None:
                    RegularClassB.__init__(self, y=y)
                self.z = z
                init_shape = kwargs.pop("shape", z.shape)
                init_device = kwargs.pop("device", z.device)
                # Call TensorAnnotated.__init__ directly to avoid MRO issues
                TensorAnnotated.__init__(self, shape=init_shape, device=init_device, **kwargs)

        instance = MultipleInheritanceWithTensorAnnotated(
            x=tensor_data['x'],
            y=tensor_data['y'],
            z=tensor_data['z']
        )

        # Verify original instance has all attributes
        assert hasattr(instance, 'x')
        assert hasattr(instance, 'y')
        assert hasattr(instance, 'z')

        instance_restored = pytree_roundtrip(instance)

        # Only the TensorAnnotated subclass attribute should be preserved
        assert_attribute_preserved(instance, instance_restored, 'z')

        # Attributes from regular classes should be missing
        assert_attribute_missing(instance_restored, 'x')
        assert_attribute_missing(instance_restored, 'y')

    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4), (2, 3, 4, 5)])
    def test_shape_consistency_in_inheritance(self, shape):
        """Tests that shape validation works correctly in inheritance scenarios.

        Validates that:
        - Shape consistency is enforced across inherited tensors
        - Different batch shapes are properly validated
        - Shape preservation works through pytree operations
        """
        class ShapeConsistentChild(BaseTensorAnnotatedA):
            y: Tensor

            def __init__(self, x: Tensor, y: Tensor, **kwargs):
                # Validate shapes match
                if x.shape != y.shape:
                    raise ValueError(f"Shape mismatch: x.shape={x.shape}, y.shape={y.shape}")
                self.y = y
                super().__init__(x=x, **kwargs)

        x = torch.randn(shape)
        y = torch.randn(shape)

        instance = ShapeConsistentChild(x=x, y=y)
        instance_restored = pytree_roundtrip(instance)

        # Verify shape preservation
        assert instance_restored.x.shape == shape
        assert instance_restored.y.shape == shape
        assert_attribute_preserved(instance, instance_restored, 'x')
        assert_attribute_preserved(instance, instance_restored, 'y')


class TestSubclassingEdgeCases:
    """Tests for edge cases and complex inheritance scenarios.

    These tests cover unusual but valid inheritance patterns and ensure
    robust behavior in complex scenarios.
    """

    def test_deep_inheritance_chain(self, tensor_data):
        """Tests inheritance chains with multiple levels.

        Validates that:
        - Deep inheritance chains work correctly
        - Attributes are properly preserved through multiple inheritance levels
        - PyTree operations work with complex inheritance hierarchies
        """
        class Level1(BaseTensorAnnotatedA):
            pass

        class Level2(Level1):
            y: Tensor

            def __init__(self, x: Tensor, y: Tensor, **kwargs):
                self.y = y
                super().__init__(x=x, **kwargs)

        class Level3(Level2):
            z: Tensor

            def __init__(self, x: Tensor, y: Tensor, z: Tensor, **kwargs):
                self.z = z
                super().__init__(x=x, y=y, **kwargs)

        class Level4(Level3):
            w: Tensor

            def __init__(self, x: Tensor, y: Tensor, z: Tensor, w: Tensor, **kwargs):
                self.w = w
                super().__init__(x=x, y=y, z=z, **kwargs)

        instance = Level4(
            x=tensor_data['x'],
            y=tensor_data['y'],
            z=tensor_data['z'],
            w=torch.randn(tensor_data['shape'])
        )
        instance_restored = pytree_roundtrip(instance)

        # Verify all attributes through the inheritance chain are preserved
        assert_attribute_preserved(instance, instance_restored, 'x')
        assert_attribute_preserved(instance, instance_restored, 'y')
        assert_attribute_preserved(instance, instance_restored, 'z')
        assert_attribute_preserved(instance, instance_restored, 'w')

    def test_diamond_inheritance_pattern(self, tensor_data):
        """Tests diamond inheritance pattern with TensorAnnotated.

        Validates that:
        - Diamond inheritance works correctly
        - Method resolution order is respected
        - No duplicate attribute handling occurs
        """
        class DiamondBase(TensorAnnotated):
            base_attr: Tensor

            def __init__(self, base_attr: Tensor, **kwargs):
                self.base_attr = base_attr
                # Extract shape and device to avoid conflicts
                init_shape = kwargs.pop("shape", base_attr.shape)
                init_device = kwargs.pop("device", base_attr.device)
                super().__init__(shape=init_shape, device=init_device, **kwargs)

        class DiamondLeft(DiamondBase):
            left_attr: Tensor

            def __init__(self, base_attr: Tensor, left_attr: Tensor, **kwargs):
                self.left_attr = left_attr
                super().__init__(base_attr=base_attr, **kwargs)

        class DiamondRight(DiamondBase):
            right_attr: Tensor

            def __init__(self, base_attr: Tensor, right_attr: Tensor, **kwargs):
                self.right_attr = right_attr
                super().__init__(base_attr=base_attr, **kwargs)

        class DiamondChild(DiamondLeft, DiamondRight):
            child_attr: Tensor

            def __init__(self, base_attr: Tensor, left_attr: Tensor,
                         right_attr: Tensor, child_attr: Tensor, **kwargs):
                self.child_attr = child_attr
                self.left_attr = left_attr
                self.right_attr = right_attr
                # Use the diamond base initialization directly to avoid conflicts
                DiamondBase.__init__(self, base_attr=base_attr, **kwargs)

        instance = DiamondChild(
            base_attr=tensor_data['x'],
            left_attr=tensor_data['y'],
            right_attr=tensor_data['z'],
            child_attr=torch.randn(tensor_data['shape'])
        )
        instance_restored = pytree_roundtrip(instance)

        # Verify all attributes are preserved
        assert_attribute_preserved(instance, instance_restored, 'base_attr')
        assert_attribute_preserved(instance, instance_restored, 'left_attr')
        assert_attribute_preserved(instance, instance_restored, 'right_attr')
        assert_attribute_preserved(instance, instance_restored, 'child_attr')

    def test_empty_tensor_annotated_class(self, tensor_data):
        """Tests inheritance from empty TensorAnnotated classes.

        Validates that:
        - Empty TensorAnnotated classes work as base classes
        - Child classes can add their own attributes
        - PyTree operations work with minimal base classes
        """
        class EmptyTensorAnnotated(TensorAnnotated):
            pass

        class ChildOfEmpty(EmptyTensorAnnotated):
            x: Tensor
            y: Tensor

            def __init__(self, x: Tensor, y: Tensor, **kwargs):
                self.x = x
                self.y = y
                # Extract shape and device to avoid conflicts
                init_shape = kwargs.pop("shape", x.shape)
                init_device = kwargs.pop("device", x.device)
                super().__init__(shape=init_shape, device=init_device, **kwargs)

        instance = ChildOfEmpty(x=tensor_data['x'], y=tensor_data['y'])
        instance_restored = pytree_roundtrip(instance)

        assert_attribute_preserved(instance, instance_restored, 'x')
        assert_attribute_preserved(instance, instance_restored, 'y')
