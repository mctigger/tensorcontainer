"""
Tests for TensorDirichlet distribution.

This module contains test classes that verify:
- TensorDirichlet initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
from torch.distributions import Dirichlet

from tensorcontainer.tensor_distribution.dirichlet import TensorDirichlet
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorDirichletInitialization:
    """Tests for TensorDirichlet initialization and parameter validation."""

    @pytest.mark.parametrize(
        "concentration_shape, expected_batch_shape",
        [
            ((3,), ()),
            ((5, 3), (5,)),
            ((2, 4, 3), (2, 4)),
            ((1, 1, 3), (1, 1)),
            ((7, 5, 3), (7, 5)),
        ],
    )
    def test_broadcasting_shapes(self, concentration_shape, expected_batch_shape):
        """Test that batch_shape is correctly determined from concentration parameter."""
        concentration = torch.rand(
            concentration_shape
        ).exp()  # concentration must be positive
        td_dirichlet = TensorDirichlet(concentration=concentration)
        assert td_dirichlet.batch_shape == expected_batch_shape
        assert td_dirichlet.dist().batch_shape == expected_batch_shape

    def test_scalar_parameters(self):
        """Test initialization with scalar-like parameters."""
        # Dirichlet requires at least 1D concentration (event dimension)
        concentration = torch.tensor([2.0, 3.0, 1.0])
        td_dirichlet = TensorDirichlet(concentration=concentration)
        assert td_dirichlet.batch_shape == ()
        assert td_dirichlet.event_shape == (3,)
        assert td_dirichlet.device == concentration.device

    def test_parameter_validation_deferred_to_torch(
        self, with_distributions_validation
    ):
        """Test that parameter validation is deferred to torch.distributions.Dirichlet."""
        # Negative concentration parameters should raise an error when validation is enabled
        with pytest.raises(ValueError):
            TensorDirichlet(
                concentration=torch.tensor([[1.0, 2.0, -0.1], [2.0, 3.0, 1.0]])
            )

    def test_event_shape_consistency(self):
        """Test that event_shape is consistent with concentration parameter."""
        concentration = torch.rand(2, 4, 5).exp()
        td_dirichlet = TensorDirichlet(concentration=concentration)
        assert td_dirichlet.event_shape == (5,)
        assert td_dirichlet.batch_shape == (2, 4)


class TestTensorDirichletTensorContainerIntegration:
    """Tests for TensorDirichlet integration with TensorContainer operations."""

    @pytest.mark.parametrize("param_shape", [(3,), (5, 3), (2, 4, 3)])
    def test_compile_compatibility(self, param_shape):
        """Core operations should be compatible with torch.compile."""
        concentration = torch.rand(*param_shape).exp()  # concentration must be positive
        td_dirichlet = TensorDirichlet(concentration=concentration)

        sample = td_dirichlet.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_dirichlet, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_dirichlet, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_dirichlet, sample, fullgraph=False)

    def test_pytree_integration(self):
        """
        We use the copy method as a proxy to ensure pytree integration (e.g. unflattening)
        works correctly.
        """
        concentration = torch.rand(3, 5).exp()
        original_dist = TensorDirichlet(concentration=concentration)
        copied_dist = original_dist.copy()

        # Assert that it's a new instance
        assert copied_dist is not original_dist
        assert isinstance(copied_dist, TensorDirichlet)

        # Assert that the parameters are the same
        torch.testing.assert_close(
            original_dist.concentration, copied_dist.concentration
        )

    def test_rsample_functionality(self):
        """Test that rsample works correctly for reparameterized sampling."""
        concentration = torch.rand(2, 4, 3).exp()
        td_dirichlet = TensorDirichlet(concentration=concentration)

        # Test that rsample is available
        assert td_dirichlet.has_rsample

        # Test rsample with different sample shapes
        rsample_no_shape = td_dirichlet.rsample()
        assert rsample_no_shape.shape == (2, 4, 3)

        rsample_with_shape = td_dirichlet.rsample(torch.Size([5]))
        assert rsample_with_shape.shape == (5, 2, 4, 3)

        # Test that rsample produces valid simplex samples (sum to 1)
        samples = td_dirichlet.rsample(torch.Size([100]))
        sample_sums = samples.sum(dim=-1)
        torch.testing.assert_close(
            sample_sums, torch.ones_like(sample_sums), rtol=1e-5, atol=1e-5
        )

    def test_device_consistency(self):
        """Test that distribution maintains device consistency."""
        if torch.cuda.is_available():
            concentration_cpu = torch.rand(3, 5).exp()
            concentration_cuda = concentration_cpu.cuda()

            td_cpu = TensorDirichlet(concentration=concentration_cpu)
            td_cuda = TensorDirichlet(concentration=concentration_cuda)

            assert td_cpu.device == torch.device("cpu")
            assert td_cuda.device is not None and td_cuda.device.type == "cuda"

            # Test that samples are on the correct device
            sample_cpu = td_cpu.sample()
            sample_cuda = td_cuda.sample()

            assert sample_cpu.device == torch.device("cpu")
            assert sample_cuda.device.type == "cuda"


class TestTensorDirichletAPIMatch:
    """
    Tests that the TensorDirichlet API matches the PyTorch Dirichlet API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorDirichlet matches
        torch.distributions.Dirichlet.
        """
        assert_init_signatures_match(TensorDirichlet, Dirichlet)

    def test_properties_match(self):
        """
        Tests that the properties of TensorDirichlet match
        torch.distributions.Dirichlet.
        """
        assert_properties_signatures_match(TensorDirichlet, Dirichlet)

    def test_property_values_match(self):
        """
        Tests that the property values of TensorDirichlet match
        torch.distributions.Dirichlet.
        """
        concentration = torch.rand(3, 5).exp()
        td_dirichlet = TensorDirichlet(concentration=concentration)
        assert_property_values_match(td_dirichlet)

    def test_sample_shape_consistency(self):
        """Test that sample shapes match between TensorDirichlet and Dirichlet."""
        concentration = torch.rand(2, 4, 3).exp()
        td_dirichlet = TensorDirichlet(concentration=concentration)
        torch_dirichlet = Dirichlet(concentration=concentration)

        # Test sample without shape
        td_sample = td_dirichlet.sample()
        torch_sample = torch_dirichlet.sample()
        assert td_sample.shape == torch_sample.shape

        # Test sample with shape
        sample_shape = (5, 2)
        td_sample_shaped = td_dirichlet.sample(torch.Size(sample_shape))
        torch_sample_shaped = torch_dirichlet.sample(sample_shape)
        assert td_sample_shaped.shape == torch_sample_shaped.shape

    def test_log_prob_consistency(self):
        """Test that log_prob values are consistent between implementations."""
        concentration = torch.rand(2, 3).exp()
        td_dirichlet = TensorDirichlet(concentration=concentration)
        torch_dirichlet = Dirichlet(concentration=concentration)

        # Generate a valid simplex sample
        sample = torch_dirichlet.sample()

        td_log_prob = td_dirichlet.log_prob(sample)
        torch_log_prob = torch_dirichlet.log_prob(sample)

        torch.testing.assert_close(td_log_prob, torch_log_prob, rtol=1e-5, atol=1e-5)

    def test_entropy_consistency(self):
        """Test that entropy values are consistent between implementations."""
        concentration = torch.rand(2, 4, 3).exp()
        td_dirichlet = TensorDirichlet(concentration=concentration)
        torch_dirichlet = Dirichlet(concentration=concentration)

        td_entropy = td_dirichlet.entropy()
        torch_entropy = torch_dirichlet.entropy()

        torch.testing.assert_close(td_entropy, torch_entropy, rtol=1e-5, atol=1e-5)

    def test_mean_and_variance_consistency(self):
        """Test that mean and variance are consistent between implementations."""
        concentration = torch.rand(2, 5).exp()
        td_dirichlet = TensorDirichlet(concentration=concentration)
        torch_dirichlet = Dirichlet(concentration=concentration)

        # Test mean
        td_mean = td_dirichlet.mean
        torch_mean = torch_dirichlet.mean
        torch.testing.assert_close(td_mean, torch_mean, rtol=1e-5, atol=1e-5)

        # Test variance
        td_variance = td_dirichlet.variance
        torch_variance = torch_dirichlet.variance
        torch.testing.assert_close(td_variance, torch_variance, rtol=1e-5, atol=1e-5)

    def test_batch_and_event_shape_consistency(self):
        """Test that batch_shape and event_shape are consistent."""
        concentration = torch.rand(3, 2, 4).exp()
        td_dirichlet = TensorDirichlet(concentration=concentration)
        torch_dirichlet = Dirichlet(concentration=concentration)

        assert td_dirichlet.batch_shape == torch_dirichlet.batch_shape
        assert td_dirichlet.event_shape == torch_dirichlet.event_shape
        assert td_dirichlet.has_rsample == torch_dirichlet.has_rsample
