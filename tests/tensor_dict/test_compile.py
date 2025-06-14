import pytest
import torch

from rtd.tensor_dict import TensorDict
from tests.tensor_dict.compile_utils import assert_td_equal, run_and_compare_compiled


@pytest.fixture
def base_td_data():
    """Provides raw data for a simple TensorDict."""
    return {
        "obs": torch.randn(4, 5, 3, 32, 32),
        "reward": torch.randn(4, 5, 1),
    }


@pytest.fixture
def simple_td(base_td_data):
    """A simple TensorDict instance."""
    return TensorDict(base_td_data, shape=(4, 5), device="cpu")


@pytest.fixture
def nested_td():
    """A TensorDict with nested structures."""
    B, T = 2, 3
    return TensorDict(
        {
            "obs": torch.randn(B, T, 10),
            "nested": TensorDict(
                {"state": torch.randn(B, T, 5)}, shape=(B, T), device="cpu"
            ),
        },
        shape=(B, T),
        device="cpu",
    )


class TestTensorDictCompilation:
    """
    Tests for TensorDict operations under torch.compile.
    """

    def test_creation_from_tensors_in_compiled_function(self, base_td_data):
        """
        Verifies that a TensorDict can be successfully created from raw tensors
        within a torch.compile'd function.
        """

        def create_td_from_tensors(obs_arg, reward_arg):
            td = TensorDict(
                {"obs": obs_arg, "reward": reward_arg},
                shape=(4, 5),
                device="cpu",
            )
            return td

        obs, reward = base_td_data["obs"], base_td_data["reward"]
        run_and_compare_compiled(create_td_from_tensors, obs, reward)

    def test_creation_from_nested_dict_in_compiled_function(self, base_td_data):
        """
        Verifies that a TensorDict can be successfully created from a nested dictionary
        within a torch.compile'd function.
        """

        def create_td_from_nested_dict(obs_arg, reward_arg):
            td = TensorDict(
                {"my_dict": {"obs": obs_arg, "reward": reward_arg}},
                shape=(4, 5),
                device="cpu",
            )
            return td

        obs, reward = base_td_data["obs"], base_td_data["reward"]
        run_and_compare_compiled(create_td_from_nested_dict, obs, reward)

    def test_setting_values_in_compiled_function(self, simple_td):
        """
        Tests that setting and modifying values within a TensorDict
        works correctly inside a compiled function.
        """

        def modify_reward_in_td(td):
            td["reward"] = td["reward"] * 2.0
            return td["reward"]

        compiled_fn = torch.compile(modify_reward_in_td, fullgraph=True)

        # Clone to ensure independent eager and compiled runs
        simple_td_eager = simple_td.clone()
        simple_td_compiled = simple_td.clone()

        eager_result = modify_reward_in_td(simple_td_eager)
        compiled_result = compiled_fn(simple_td_compiled)

        # Compare the returned tensor directly
        assert torch.allclose(eager_result, compiled_result)
        # Also compare the modified TensorDicts
        assert_td_equal(simple_td_eager, simple_td_compiled)

    def test_stacking_tensordicts_in_compiled_function(self, simple_td):
        """
        Tests torch.stack operation on TensorDicts within a compiled function.
        """
        td_a = simple_td.clone()
        td_b = simple_td.clone()
        td_b["reward"] += 1.0  # Make it different for stacking

        def stack_tensordicts(d1, d2):
            return torch.stack([d1, d2], dim=0)

        eager_result, compiled_result = run_and_compare_compiled(
            stack_tensordicts, td_a, td_b
        )
        assert eager_result.shape == (2, 4, 5)

    def test_concatenating_tensordicts_in_compiled_function(self, simple_td):
        """
        Tests torch.cat operation on TensorDicts within a compiled function.
        """
        td_a = simple_td.clone()
        td_b = simple_td.clone()

        def concatenate_tensordicts(d1, d2):
            return torch.cat([d1, d2], dim=1)

        eager_result, compiled_result = run_and_compare_compiled(
            concatenate_tensordicts, td_a, td_b
        )
        assert eager_result.shape == (4, 10)

    @pytest.mark.parametrize(
        "index",
        [
            0,
            slice(1, 3),
            torch.tensor([0, 3]),  # advanced indexing
        ],
    )
    def test_indexing_tensordict_in_compiled_function(self, simple_td, index):
        """
        Tests various forms of indexing a TensorDict within a compiled function.
        """

        def index_tensordict(td_arg):
            return td_arg[index]

        run_and_compare_compiled(index_tensordict, simple_td)

    def test_operations_on_nested_tensordict_in_compiled_function(self, nested_td):
        """
        Tests slicing and modifying a nested TensorDict within a compiled function.
        """

        def operate_on_nested_td(td):
            sliced_td = td[0]
            sliced_td["nested"]["state"] = sliced_td["nested"]["state"] + 5.0
            return sliced_td

        compiled_fn = torch.compile(operate_on_nested_td, fullgraph=True)

        # Clone to ensure independent eager and compiled runs
        nested_td_eager = nested_td.clone()
        nested_td_compiled = nested_td.clone()

        eager_result = operate_on_nested_td(nested_td_eager)
        compiled_result = compiled_fn(nested_td_compiled)

        assert eager_result.shape == (3,)
        assert eager_result["nested"].shape == (3,)
        assert_td_equal(eager_result, compiled_result)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="This test requires a CUDA device"
    )
    def test_device_move_in_compiled_function(self, simple_td):
        """
        Tests moving a TensorDict to a CUDA device within a compiled function.
        """

        def move_td_to_cuda(td_arg):
            return td_arg.to("cuda")

        eager_result, compiled_result = run_and_compare_compiled(
            move_td_to_cuda, simple_td
        )
        assert eager_result.device == "cuda"

    def test_dtype_change_in_compiled_function(self, simple_td):
        """
        Tests changing the dtype of a TensorDict's tensors within a compiled function.
        """

        def change_td_dtype(td_arg):
            return td_arg.to(torch.float16)

        eager_result, compiled_result = run_and_compare_compiled(
            change_td_dtype, simple_td
        )
        assert eager_result["obs"].dtype == torch.float16
