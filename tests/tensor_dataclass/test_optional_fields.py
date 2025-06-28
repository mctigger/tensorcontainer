import pytest
import torch

from tests.conftest import skipif_no_compile
from tests.tensor_dataclass.conftest import OptionalFieldsTestClass


class TestOptionalFields:
    @pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
    def test_optional_tensor_is_none(self, optional_fields_instance, execution_mode):
        """
        Tests that TensorDataclass can be initialized and stacked
        when an Optional[Tensor] field is None.
        This should pass after the validation fix.
        """

        def _test_stack_none():
            data1 = OptionalFieldsTestClass(
                shape=(4,),
                device=None,
                obs=torch.ones(4, 32, 32),
                reward=None,  # Optional field is None
                info=["step1"],
            )

            data2 = OptionalFieldsTestClass(
                shape=(4,),
                device=None,
                obs=torch.ones(4, 32, 32),
                reward=None,
                info=["step1"],
            )

            # Stacking involves flatten/unflatten and __post_init__ validation
            # Before the fix, this would raise AttributeError in _tree_validate_shape/_device
            stacked_data = torch.stack([data1, data2], dim=0)
            return stacked_data

        if execution_mode == "compiled":
            from tests.compile_utils import run_and_compare_compiled

            run_and_compare_compiled(_test_stack_none)
        else:
            stacked_data = _test_stack_none()
            assert stacked_data.obs.shape == (2, 4, 32, 32)
            assert stacked_data.reward is None
            assert stacked_data.info == [
                "step1"
            ]  # PyTree takes metadata from the first element
            assert stacked_data.optional_meta is None
            assert stacked_data.optional_meta_val == "value"
            assert stacked_data.shape == (2, 4)
            # When device is None for inputs, stacked output device can also be None or inferred.
            # If all input tensors are on CPU (default for ones if device not given), stacked device will be CPU.
            # If inputs have device=None, and tensors are created without specific device, they are CPU.
            # The TensorDataclass device itself is None initially, but __post_init__ infers it from children.
            assert stacked_data.device == torch.device("cpu")

    @pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
    def test_optional_tensor_is_tensor(self, optional_fields_instance, execution_mode):
        """
        Tests that TensorDataclass can be initialized and stacked
        when an Optional[Tensor] field is an actual Tensor.
        """

        def _test_stack_tensor():
            data1 = OptionalFieldsTestClass(
                shape=(4,),
                device=None,  # Explicitly setting device to None for the container
                obs=torch.ones(4, 32, 32),  # Tensor will be on CPU by default
                reward=torch.ones(4),  # Optional field is a Tensor (on CPU by default)
            )

            data2 = OptionalFieldsTestClass(
                shape=(4,),
                device=None,
                obs=torch.ones(4, 32, 32),
                reward=torch.ones(4) * 2,
            )

            stacked_data = torch.stack([data1, data2], dim=0)
            return stacked_data

        if execution_mode == "compiled":
            from tests.compile_utils import run_and_compare_compiled

            run_and_compare_compiled(_test_stack_tensor)
        else:
            stacked_data = _test_stack_tensor()
            assert stacked_data.obs.shape == (2, 4, 32, 32)
            assert stacked_data.reward is not None
            assert stacked_data.reward.shape == (2, 4)
            assert torch.equal(stacked_data.reward[0], torch.ones(4))
            assert torch.equal(stacked_data.reward[1], torch.ones(4) * 2)
            assert stacked_data.info == []  # default_factory creates new list for data1
            assert stacked_data.optional_meta is None
            assert stacked_data.optional_meta_val == "value"
            assert stacked_data.shape == (2, 4)
            # Device of stacked_data should be CPU as underlying tensors are on CPU
            assert stacked_data.device == torch.device("cpu")

    @pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
    def test_default_factory_field(self, optional_fields_instance, execution_mode):
        """
        Tests that fields with default_factory are handled correctly (as metadata).
        """

        def _test_default_factory():
            data = OptionalFieldsTestClass(
                shape=(4,), device=None, obs=torch.ones(4, 5), reward=None
            )
            cloned_data = data.clone()
            return data, cloned_data

        if execution_mode == "compiled":
            from tests.compile_utils import run_and_compare_compiled

            run_and_compare_compiled(_test_default_factory)
        else:
            data, cloned_data = _test_default_factory()
            assert data.info == []  # default_factory called
            assert cloned_data.info == []  # metadata copied
            # For immutable types like list, dataclasses.replace (used by clone indirectly) might create a new list.
            # If it's a shallow copy, it might be the same. Let's check for value equality.
            assert cloned_data.info is data.info

    @pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
    def test_validation_with_none_field_direct_init(
        self, optional_fields_instance, execution_mode
    ):
        """
        Tests direct initialization that would trigger __post_init__ validation.
        This test is crucial for the bug.
        """

        def _test_init_none():
            instance = OptionalFieldsTestClass(
                shape=(4,),
                device=None,
                obs=torch.ones(4, 10),
                reward=None,  # This is the key: an Optional[Tensor] that is None
            )
            return instance

        if execution_mode == "compiled":
            from tests.compile_utils import run_and_compare_compiled

            run_and_compare_compiled(_test_init_none)
        else:
            try:
                instance = _test_init_none()
                assert instance.obs.shape == (4, 10)
                assert instance.reward is None
                assert instance.shape == (4,)
                assert instance.device == torch.device(
                    "cpu"
                )  # Inferred from obs tensor
            except AttributeError as e:
                pytest.fail(
                    f"AttributeError during initialization with None tensor: {e}"
                )

    @pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
    def test_validation_with_all_fields_present(
        self, optional_fields_instance, execution_mode
    ):
        """
        Tests direct initialization when all fields, including optional tensor, are present.
        """

        def _test_init_all_fields():
            instance = OptionalFieldsTestClass(
                shape=(4,),
                device=None,  # Container device
                obs=torch.ones(4, 10),  # on CPU
                reward=torch.zeros(4, 2),  # Reward is a tensor, on CPU
            )
            return instance

        if execution_mode == "compiled":
            from tests.compile_utils import run_and_compare_compiled

            run_and_compare_compiled(_test_init_all_fields)
        else:
            try:
                instance = _test_init_all_fields()
                assert instance.obs.shape == (4, 10)
                assert instance.reward is not None
                assert instance.reward.shape == (4, 2)
                assert instance.shape == (4,)
                assert instance.device == torch.device("cpu")  # Inferred from tensors
            except AttributeError as e:
                pytest.fail(
                    f"AttributeError during initialization with present tensor: {e}"
                )

    def test_validation_device_mismatch_with_none(self, optional_fields_instance):
        """
        Tests that device validation still works correctly even if an optional field is None.
        The presence of a None field should not bypass other validations.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        with pytest.raises(ValueError, match="Device mismatch"):
            OptionalFieldsTestClass(
                shape=(4,),
                device=torch.device("cpu"),  # TensorDataclass device
                obs=torch.ones(4, 10, device="cuda"),  # Tensor on different device
                reward=None,  # Optional field is None
            )

    def test_validation_shape_mismatch_with_none(self, optional_fields_instance):
        """
        Tests that shape validation still works correctly even if an optional field is None.
        """
        with pytest.raises(ValueError, match="Shape mismatch"):
            OptionalFieldsTestClass(
                shape=(4,),  # TensorDataclass shape (batch_size=4)
                device=None,
                obs=torch.ones(3, 10),  # Tensor with incompatible batch_size
                reward=None,  # Optional field is None
            )

    @skipif_no_compile
    @pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
    def test_default_factory_tensor_field(
        self, optional_fields_instance, execution_mode
    ):
        """
        Tests that TensorDataclass can handle default_factory fields with torch.Tensor values.
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

            stacked_data = torch.stack([data1, data2], dim=0)
            return data1, stacked_data

        if execution_mode == "compiled":
            from tests.compile_utils import run_and_compare_compiled

            run_and_compare_compiled(_test_default_factory_tensor)
        else:
            data1, stacked_data = _test_default_factory_tensor()
            # default_tensor should be initialized by default_factory
            assert data1.default_tensor.shape == (4,)
            assert torch.equal(data1.default_tensor, torch.zeros(4))

            # default_tensor should be stacked correctly
            assert stacked_data.default_tensor.shape == (2, 4)
            assert torch.equal(stacked_data.default_tensor, torch.zeros(2, 4))
            assert stacked_data.device == torch.device("cpu")
