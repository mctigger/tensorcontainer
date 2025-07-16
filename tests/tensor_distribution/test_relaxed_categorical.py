import pytest
import torch
from torch.distributions import RelaxedOneHotCategorical as TorchRelaxedCategorical

from tensorcontainer.tensor_distribution.relaxed_categorical import (
    TensorRelaxedCategorical,
)
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorRelaxedCategoricalInitialization:
    @pytest.mark.parametrize(
        "temperature_val, param_type, param_shape, expected_batch_shape",
        [
            (0.5, "probs", (5,), (5,)),
            (0.5, "logits", (3, 5), (3,)),
            (torch.tensor(0.5), "probs", (2, 4, 5), (2, 4)),
            (torch.tensor([0.5, 0.8]), "logits", (2, 5), (2,)),
        ],
    )
    def test_valid_initialization(
        self, temperature_val, param_type, param_shape, expected_batch_shape
    ):
        """Test valid initializations with various parameter types and shapes."""
        if param_type == "probs":
            param = torch.rand(*param_shape)
            dist = TensorRelaxedCategorical(temperature=temperature_val, probs=param)
        else:
            param = torch.randn(*param_shape)
            dist = TensorRelaxedCategorical(temperature=temperature_val, logits=param)

        assert isinstance(dist, TensorRelaxedCategorical)
        assert dist.batch_shape == expected_batch_shape
        assert dist.dist().batch_shape == expected_batch_shape

    def test_init_no_params_raises_error(self):
        """A RuntimeError should be raised when neither probs nor logits are provided."""
        with pytest.raises(
            RuntimeError, match="Either 'probs' or 'logits' must be provided."
        ):
            TensorRelaxedCategorical(temperature=torch.tensor(0.5))

    def test_init_no_temperature_raises_error(self):
        """A RuntimeError should be raised when temperature is not provided."""
        with pytest.raises(RuntimeError, match="'temperature' must be provided."):
            TensorRelaxedCategorical(temperature=None, probs=torch.rand(5))


class TestTensorRelaxedCategoricalTensorContainerIntegration:
    @pytest.mark.parametrize("param_shape", [(5,), (3, 5), (2, 4, 5)])
    @pytest.mark.parametrize("param_type", ["probs", "logits"])
    def test_compile_compatibility(self, param_shape, param_type):
        """Core operations should be compatible with torch.compile."""
        temperature = torch.tensor(0.5, requires_grad=True)
        if param_type == "probs":
            param = torch.rand(*param_shape, requires_grad=True)
            td_dist = TensorRelaxedCategorical(temperature=temperature, probs=param)
        else:
            param = torch.randn(*param_shape, requires_grad=True)
            td_dist = TensorRelaxedCategorical(temperature=temperature, logits=param)

        sample = td_dist.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_dist, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_dist, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_dist, sample, fullgraph=False)

    @pytest.mark.parametrize("param_type", ["probs", "logits"])
    def test_copy_method(self, param_type):
        """Test that the .copy() method works correctly."""
        temperature = torch.tensor(0.5)
        if param_type == "probs":
            param = torch.rand(3, 5)
            original_dist = TensorRelaxedCategorical(
                temperature=temperature, probs=param
            )
        else:
            param = torch.randn(3, 5)
            original_dist = TensorRelaxedCategorical(
                temperature=temperature, logits=param
            )

        copied_dist = original_dist.copy()

        assert copied_dist is not original_dist
        assert isinstance(copied_dist, TensorRelaxedCategorical)
        torch.testing.assert_close(original_dist.temperature, copied_dist.temperature)
        if param_type == "probs":
            torch.testing.assert_close(original_dist.probs, copied_dist.probs)
        else:
            torch.testing.assert_close(original_dist.logits, copied_dist.logits)


class TestTensorRelaxedCategoricalAPIMatch:
    """
    Tests that the TensorRelaxedCategorical API matches the PyTorch RelaxedOneHotCategorical API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorRelaxedCategorical matches
        torch.distributions.RelaxedOneHotCategorical.
        """
        assert_init_signatures_match(TensorRelaxedCategorical, TorchRelaxedCategorical)

    def test_properties_match(self):
        """
        Tests that the properties of TensorRelaxedCategorical match
        torch.distributions.RelaxedOneHotCategorical.
        """
        assert_properties_signatures_match(
            TensorRelaxedCategorical, TorchRelaxedCategorical
        )

    @pytest.mark.parametrize("param_type", ["probs", "logits"])
    def test_property_values_match(self, param_type):
        """
        Tests that the property values of TensorRelaxedCategorical match
        torch.distributions.RelaxedOneHotCategorical.
        """
        temperature = torch.tensor(0.5)
        if param_type == "probs":
            param = torch.rand(3, 5)
            td_dist = TensorRelaxedCategorical(temperature=temperature, probs=param)
        else:
            param = torch.randn(3, 5)
            td_dist = TensorRelaxedCategorical(temperature=temperature, logits=param)
        assert_property_values_match(td_dist)
