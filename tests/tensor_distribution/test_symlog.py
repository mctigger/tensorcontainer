"""
Tests for TensorSymLog distribution.

This module contains test classes that verify:
- TensorSymLog initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch

from tensorcontainer.distributions.symlog import SymLogDistribution
from tensorcontainer.tensor_distribution.symlog import TensorSymLog
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorSymLogInitialization:
    @pytest.mark.parametrize(
        "loc_shape, scale_shape, expected_batch_shape",
        [
            ((), (), ()),
            ((5,), (), (5,)),
            ((), (5,), (5,)),
            ((3, 5), (5,), (3, 5)),
            ((5,), (3, 5), (3, 5)),
            ((2, 4, 5), (5,), (2, 4, 5)),
            ((5,), (2, 4, 5), (2, 4, 5)),
            ((2, 4, 5), (2, 4, 5), (2, 4, 5)),
        ],
    )
    def test_broadcasting_shapes(self, loc_shape, scale_shape, expected_batch_shape):
        """Test that batch_shape is correctly determined by broadcasting."""
        loc = torch.randn(loc_shape)
        scale = torch.rand(scale_shape).exp()  # scale must be positive
        td_symlog = TensorSymLog(loc=loc, scale=scale)
        assert td_symlog.batch_shape == expected_batch_shape
        assert td_symlog.dist().batch_shape == expected_batch_shape

    def test_initialization_with_scalars(self):
        """Test initialization with scalar parameters."""
        td_symlog = TensorSymLog(loc=0.0, scale=1.0)
        assert td_symlog.batch_shape == torch.Size(())
        assert td_symlog.loc.shape == torch.Size(())
        assert td_symlog.scale.shape == torch.Size(())

    def test_initialization_with_tensors(self):
        """Test initialization with tensor parameters."""
        loc = torch.tensor([1.0, -2.0, 0.0])
        scale = torch.tensor([0.5, 1.0, 2.0])
        td_symlog = TensorSymLog(loc=loc, scale=scale)
        assert td_symlog.batch_shape == torch.Size([3])
        assert torch.allclose(td_symlog.loc, loc)
        assert torch.allclose(td_symlog.scale, scale)


class TestTensorSymLogTensorContainerIntegration:
    @pytest.mark.parametrize("param_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, param_shape):
        """Core operations should be compatible with torch.compile."""
        loc = torch.randn(*param_shape)
        scale = torch.rand(*param_shape).exp()  # scale must be positive
        td_symlog = TensorSymLog(loc=loc, scale=scale)

        sample = td_symlog.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_symlog, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_symlog, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_symlog, sample, fullgraph=False)

    def test_device_compatibility(self):
        """Test that the distribution works on different devices."""
        loc = torch.tensor([1.0, -2.0, 0.0])
        scale = torch.tensor([0.5, 1.0, 2.0])
        td_symlog = TensorSymLog(loc=loc, scale=scale)

        # Test CPU
        sample = td_symlog.sample()
        assert sample.device == loc.device

        # Test GPU if available
        if torch.cuda.is_available():
            loc_gpu = loc.cuda()
            scale_gpu = scale.cuda()
            td_symlog_gpu = TensorSymLog(loc=loc_gpu, scale=scale_gpu)
            sample_gpu = td_symlog_gpu.sample()
            assert sample_gpu.is_cuda


class TestTensorSymLogAPIMatch:
    """
    Tests that the TensorSymLog API matches the SymLogDistribution API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorSymLog matches
        SymLogDistribution.
        """
        assert_init_signatures_match(TensorSymLog, SymLogDistribution)

    def test_properties_match(self):
        """
        Tests that the properties of TensorSymLog match
        SymLogDistribution.
        """
        assert_properties_signatures_match(TensorSymLog, SymLogDistribution)

    def test_property_values_match(self):
        """
        Tests that the property values of TensorSymLog match
        SymLogDistribution.
        """
        loc = torch.tensor([1.0, -2.0, 0.0])
        scale = torch.tensor([0.5, 1.0, 2.0])
        td_symlog = TensorSymLog(loc=loc, scale=scale)
        assert_property_values_match(td_symlog)

    def test_distribution_equivalence(self):
        """
        Tests that TensorSymLog produces the same results as SymLogDistribution.
        """
        loc = torch.tensor([1.0, -2.0, 0.0])
        scale = torch.tensor([0.5, 1.0, 2.0])

        # Create both distributions
        td_symlog = TensorSymLog(loc=loc, scale=scale)
        symlog_dist = SymLogDistribution(loc=loc, scale=scale)

        # Test sampling
        torch.manual_seed(42)
        td_sample = td_symlog.sample(torch.Size([100]))
        torch.manual_seed(42)
        symlog_sample = symlog_dist.sample(torch.Size([100]))
        assert torch.allclose(td_sample, symlog_sample, rtol=1e-5, atol=1e-5)

        # Test log_prob
        test_values = torch.tensor([0.0, 1.0, -1.0])
        td_log_prob = td_symlog.log_prob(test_values)
        symlog_log_prob = symlog_dist.log_prob(test_values)
        assert torch.allclose(td_log_prob, symlog_log_prob, rtol=1e-5, atol=1e-5)

        # Test mean and mode
        assert torch.allclose(td_symlog.mean, symlog_dist.mean, rtol=1e-5, atol=1e-5)
        assert torch.allclose(td_symlog.mode, symlog_dist.mode, rtol=1e-5, atol=1e-5)


class TestTensorSymLogFunctionality:
    """Test specific functionality of TensorSymLog."""

    @pytest.fixture
    def sample_params(self):
        """Common test parameters."""
        return {
            "loc": torch.tensor([1.0, -2.0, 0.0]),
            "scale": torch.tensor([0.5, 1.0, 2.0]),
        }

    @pytest.fixture
    def sample_distribution(self, sample_params):
        """Common test distribution."""
        return TensorSymLog(sample_params["loc"], sample_params["scale"])

    def test_sampling(self, sample_distribution):
        """Test sampling functionality."""
        # Test default sampling
        sample = sample_distribution.sample()
        assert sample.shape == sample_distribution.batch_shape
        assert torch.all(torch.isfinite(sample))

        # Test sampling with specific shape
        sample_shape = torch.Size([10, 2])
        samples = sample_distribution.sample(sample_shape)
        assert samples.shape == sample_shape + sample_distribution.batch_shape
        assert torch.all(torch.isfinite(samples))

    def test_reparameterized_sampling(self, sample_distribution):
        """Test reparameterized sampling functionality."""
        # Check if the distribution supports rsample
        assert sample_distribution.has_rsample == sample_distribution.dist().has_rsample

        # SymLogDistribution claims to support rsample but doesn't actually
        # provide gradients due to the non-affine transform
        assert sample_distribution.has_rsample

        # Test rsample - it should work but without gradients
        rsample = sample_distribution.rsample()
        assert rsample.shape == sample_distribution.batch_shape
        assert torch.all(torch.isfinite(rsample))
        # The rsample should not have requires_grad=True due to non-affine transform
        assert not rsample.requires_grad

        # Test rsample with specific shape
        sample_shape = torch.Size([10, 2])
        rsamples = sample_distribution.rsample(sample_shape)
        assert rsamples.shape == sample_shape + sample_distribution.batch_shape
        assert torch.all(torch.isfinite(rsamples))
        assert not rsamples.requires_grad

    def test_log_prob(self, sample_distribution):
        """Test log probability computation."""
        # Test log_prob at mode (should be high probability)
        mode_log_prob = sample_distribution.log_prob(sample_distribution.mode)
        assert mode_log_prob.shape == sample_distribution.batch_shape
        assert torch.all(torch.isfinite(mode_log_prob))

        # Test log_prob for random values
        test_values = torch.tensor([0.0, 1.0, -1.0])
        log_probs = sample_distribution.log_prob(test_values)
        assert log_probs.shape == test_values.shape
        assert torch.all(torch.isfinite(log_probs))

    def test_entropy(self, sample_distribution):
        """Test entropy computation."""
        # SymLogDistribution doesn't implement entropy
        with pytest.raises(NotImplementedError):
            sample_distribution.entropy()

    def test_mean_and_variance(self, sample_distribution):
        """Test mean and variance properties."""
        mean = sample_distribution.mean

        assert mean.shape == sample_distribution.batch_shape
        assert torch.all(torch.isfinite(mean))

        # SymLogDistribution doesn't implement variance
        with pytest.raises(NotImplementedError):
            sample_distribution.variance

    def test_mode_property(self, sample_distribution):
        """Test mode property."""
        mode = sample_distribution.mode
        assert mode.shape == sample_distribution.batch_shape
        assert torch.all(torch.isfinite(mode))

    def test_batch_and_event_shape(self, sample_distribution):
        """Test batch_shape and event_shape properties."""
        assert sample_distribution.batch_shape == sample_distribution.loc.shape
        assert sample_distribution.event_shape == torch.Size()  # Scalar event

    def test_support_property(self, sample_distribution):
        """Test support property."""
        support = sample_distribution.support
        # SymLogDistribution has real support
        assert support is not None

    def test_cdf_and_icdf(self, sample_distribution):
        """Test CDF and inverse CDF functionality."""
        # Test CDF
        test_values = torch.tensor([0.0, 1.0, -1.0])
        cdf_values = sample_distribution.cdf(test_values)

        assert cdf_values.shape == test_values.shape
        assert torch.all((cdf_values >= 0) & (cdf_values <= 1))
        assert torch.all(torch.isfinite(cdf_values))

        # Test ICDF
        prob_values = torch.tensor([0.1, 0.5, 0.9])
        icdf_values = sample_distribution.icdf(prob_values)

        assert icdf_values.shape == prob_values.shape
        assert torch.all(torch.isfinite(icdf_values))

        # Test CDF/ICDF inverse relationship
        reconstructed_probs = sample_distribution.cdf(icdf_values)
        assert torch.allclose(reconstructed_probs, prob_values, atol=1e-5)

    def test_unflatten_distribution(self, sample_params):
        """Test _unflatten_distribution class method."""
        td_symlog = TensorSymLog(sample_params["loc"], sample_params["scale"])

        # Get attributes
        attributes = {
            "_loc": td_symlog._loc,
            "_scale": td_symlog._scale,
            "_validate_args": td_symlog._validate_args,
        }

        # Reconstruct distribution
        reconstructed = TensorSymLog._unflatten_distribution(attributes)

        # Check that the reconstructed distribution is equivalent
        assert torch.allclose(reconstructed.loc, td_symlog.loc)
        assert torch.allclose(reconstructed.scale, td_symlog.scale)
        assert reconstructed._validate_args == td_symlog._validate_args
