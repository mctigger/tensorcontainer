"""
Tests for TensorDataclass with optional and default fields.

This module contains test classes that verify the correct handling of
Optional[torch.Tensor] fields and fields with default_factory in TensorDataClass
instances, including initialization, stacking, and cloning behaviors.
"""

import dataclasses

import pytest
import torch
from typing import cast

from tests.conftest import skipif_no_compile
from tests.tensor_dataclass.conftest import OptionalFieldsTestClass
from tests.compile_utils import run_and_compare_compiled
from tensorcontainer.tensor_dataclass import TensorDataClass


class InitFalseTestClass(TensorDataClass):
    a: torch.Tensor
    b: torch.Tensor = dataclasses.field(init=False)
    c: str = dataclasses.field(init=False, default="hello")

    def __post_init__(self):
        super().__post_init__()
        # b must be initialized here based on other fields
        if hasattr(self, "a"):
            self.b = self.a + 1


class TestOptionalTensorInitialization:
    """
    Tests the initialization logic of TensorDataclass with optional fields.

    This suite verifies that:
    - The dataclass can be initialized when an Optional[Tensor] field is None.
    - The dataclass can be initialized when an Optional[Tensor] field is a Tensor.
    - Shape validation correctly raises an error when a tensor has an incompatible batch size, even with optional fields.
    """

    def test_init_with_optional_tensor_as_none(self):
        """
        Tests that the dataclass can be initialized when an Optional[Tensor]
        field is None.
        """

        def _test_init_none():
            return OptionalFieldsTestClass(
                shape=(4,),
                device=None,
                obs=torch.ones(4, 10),
                reward=None,
            )

        instance, _ = run_and_compare_compiled(_test_init_none)
        assert instance.obs.shape == (4, 10)
        assert instance.reward is None
        assert instance.shape == (4,)

    def test_init_with_optional_tensor_as_tensor(self):
        """
        Tests that the dataclass can be initialized when an Optional[Tensor]
        field is a Tensor.
        """

        def _test_init_all_fields():
            return OptionalFieldsTestClass(
                shape=(4,),
                device=None,
                obs=torch.ones(4, 10),
                reward=torch.zeros(4, 2),
            )

        instance, _ = run_and_compare_compiled(_test_init_all_fields)
        assert instance.obs.shape == (4, 10)
        assert instance.reward is not None
        assert instance.reward.shape == (4, 2)
        assert instance.shape == (4,)

    def test_init_raises_error_on_shape_mismatch(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            OptionalFieldsTestClass(
                shape=(4,),
                device=None,
                obs=torch.ones(3, 10),
                reward=None,
            )


class TestDefaultValueFields:
    """
    Tests the behavior of fields defined with `default` or `default_factory` in
    TensorDataClass.

    This suite verifies that:
    - Metadata fields with `default_factory` are correctly handled during cloning.
    - Tensor fields with `default_factory` are correctly handled during stacking.
    - Using a mutable `default` for a tensor field raises a `ValueError`.
    """

    def test_default_factory_metadata_clone(self):
        """
        Tests that metadata fields with `default_factory` are correctly handled
        during cloning.
        """

        def _test_default_factory():
            data = OptionalFieldsTestClass(
                shape=(4,), device=None, obs=torch.ones(4, 5), reward=None
            )
            return data, data.clone()

        (data, cloned_data), _ = run_and_compare_compiled(_test_default_factory)
        assert data.info == []
        assert cloned_data.info == []
        assert cloned_data.info is data.info

    @skipif_no_compile
    def test_default_factory_tensor_stack(self):
        """
        Tests that tensor fields with `default_factory` are correctly handled
        during stacking.
        """

        def _test_default_factory_tensor():
            data1 = OptionalFieldsTestClass(
                shape=(4,),
                device=None,
                obs=torch.ones(4, 32, 32),
                reward=None,
                info=["step1"],
            )
            data2 = OptionalFieldsTestClass(
                shape=(4,),
                device=None,
                obs=torch.ones(4, 32, 32),
                reward=None,
                info=["step1"],
            )
            stacked_data = cast(
                OptionalFieldsTestClass,
                torch.stack([data1, data2], dim=0),  # type: ignore
            )
            return data1, stacked_data

        (data1, stacked_data), _ = run_and_compare_compiled(
            _test_default_factory_tensor
        )
        assert data1.default_tensor.shape == (4,)
        assert torch.equal(data1.default_tensor, torch.zeros(4))
        assert stacked_data.default_tensor.shape == (2, 4)
        assert torch.equal(stacked_data.default_tensor, torch.zeros(2, 4))

    @pytest.mark.xfail(
        reason="TensorDataClass does not currently validate against mutable default values."
    )
    def test_mutable_default_tensor_raises_error(self):
        """
        Tests that defining a class with a mutable default for a tensor field
        raises a ValueError, which is standard dataclass behavior.
        """
        with pytest.raises(ValueError, match="mutable default"):

            class _(TensorDataClass):
                b: torch.Tensor = torch.zeros(4)


class TestInitFalseFields:
    """
    Tests the behavior of fields with `init=False` in TensorDataClass.

    This suite verifies that:
    - Fields with `init=False` are not expected in the constructor.
    - `__post_init__` can be used to initialize `init=False` fields.
    - `init=False` fields with defaults are correctly set.
    - Stacking works correctly with `init=False` fields.
    """

    @pytest.mark.xfail(
        reason="TensorDataClass validation runs before __post_init__ can initialize the field."
    )
    def test_post_init_can_initialize_field(self):
        """
        Tests that a tensor field with init=False can be initialized in __post_init__.
        """

        def _test():
            return InitFalseTestClass(a=torch.ones(4), shape=(4,), device=None)

        instance, _ = run_and_compare_compiled(_test)
        assert torch.equal(instance.b, torch.ones(4) + 1)
        assert instance.c == "hello"
        assert instance.shape == (4,)

    def test_init_false_with_default_value(self):
        """
        Tests that a non-tensor field with init=False and a default value is
        correctly initialized.
        """

        class InitFalseDefault(TensorDataClass):
            a: torch.Tensor
            b: int = dataclasses.field(init=False, default=10)

        def _test():
            return InitFalseDefault(a=torch.ones(4), shape=(4,), device=None)

        instance, _ = run_and_compare_compiled(_test)
        assert instance.b == 10
        assert instance.shape == (4,)

    @pytest.mark.xfail(
        reason="TensorDataClass validation runs before __post_init__ can initialize the field."
    )
    @skipif_no_compile
    def test_stacking_with_init_false_field(self):
        """
        Tests that stacking works correctly with init=False fields.
        """

        def _test():
            instance1 = InitFalseTestClass(a=torch.ones(4), shape=(4,), device=None)
            instance2 = InitFalseTestClass(a=torch.zeros(4), shape=(4,), device=None)
            stacked = cast(
                InitFalseTestClass,
                torch.stack([instance1, instance2], dim=0),  # type: ignore
            )
            return stacked

        stacked, _ = run_and_compare_compiled(_test)
        assert stacked.shape == (2, 4)
        assert torch.equal(
            stacked.a, torch.stack([torch.ones(4), torch.zeros(4)], dim=0)
        )
        assert torch.equal(
            stacked.b, torch.stack([torch.ones(4) + 1, torch.zeros(4) + 1], dim=0)
        )
        assert stacked.c == "hello"  # metadata is not stacked


class TestFieldArguments:
    """
    Tests other `dataclasses.field` arguments like `repr`.

    This suite verifies that:
    - Fields with `repr=False` are excluded from the string representation.
    """

    def test_repr_false(self):
        """
        Tests that a field with repr=False is excluded from the string
        representation.
        """

        class ReprFalseTestClass(TensorDataClass):
            a: torch.Tensor
            b: torch.Tensor = dataclasses.field(repr=False)

        def _test():
            return ReprFalseTestClass(
                a=torch.ones(2), b=torch.zeros(2), shape=(2,), device=None
            )

        instance, _ = run_and_compare_compiled(_test)
        instance_repr = repr(instance)
        assert "a=" in instance_repr
        assert "b=" not in instance_repr
