"""
Tests for TensorDirac distribution.

This module contains test classes that verify:
- TensorDirac initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
from tensorcontainer.distributions import DiracDistribution
from tensorcontainer.tensor_distribution.dirac import TensorDirac
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorDiracAPIMatch:
    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorDirac matches
        DiracDistribution.
        """
        assert_init_signatures_match(TensorDirac, DiracDistribution)

    def test_properties_match(self):
        """
        Tests that the properties of TensorDirac match
        DiracDistribution.
        """
        assert_properties_signatures_match(TensorDirac, DiracDistribution)

    @pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 4, 5)])
    def test_property_values_match(self, shape):
        """
        Tests that the property values of TensorDirac match
        DiracDistribution.
        """
        value = torch.randn(*shape)
        td_dirac = TensorDirac(value=value)
        assert_property_values_match(td_dirac)

    @pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, shape):
        """Core operations should be compatible with torch.compile."""
        value = torch.randn(*shape, requires_grad=True)
        td_dirac = TensorDirac(value=value)

        def get_mean(td):
            return td.mean

        run_and_compare_compiled(get_mean, td_dirac, fullgraph=False)


class TestTensorDiracOperations:
    """Test TensorDirac distribution-specific operations."""

    @pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 4, 5)])
    def test_sample_returns_value(self, shape):
        """Sample should always return the point value."""
        value = torch.randn(*shape)
        td_dirac = TensorDirac(value=value)

        sample = td_dirac.sample()
        assert torch.allclose(sample, value)

        # Multiple samples should all be the same
        samples = td_dirac.sample((10,))
        expected = value.unsqueeze(0).expand(10, *value.shape)
        assert torch.allclose(samples, expected)

    @pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 4, 5)])
    def test_rsample_returns_value(self, shape):
        """Reparameterized sample should also return the point value."""
        value = torch.randn(*shape)
        td_dirac = TensorDirac(value=value)

        rsample = td_dirac.rsample()
        assert torch.allclose(rsample, value)

        # Multiple rsamples should all be the same
        rsamples = td_dirac.rsample((10,))
        expected = value.unsqueeze(0).expand(10, *value.shape)
        assert torch.allclose(rsamples, expected)

    @pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 4, 5)])
    def test_log_prob_exact_match(self, shape):
        """Log probability should be 0 for exact matches."""
        value = torch.randn(*shape)
        td_dirac = TensorDirac(value=value)

        log_prob = td_dirac.log_prob(value)
        assert torch.allclose(log_prob, torch.zeros_like(value))

    @pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 4, 5)])
    def test_log_prob_mismatch(self, shape):
        """Log probability should be -inf for mismatches."""
        value = torch.randn(*shape)
        td_dirac = TensorDirac(value=value)

        other_value = value + 1.0  # Different value
        log_prob = td_dirac.log_prob(other_value)
        assert torch.all(torch.isinf(log_prob))
        assert torch.all(log_prob < 0)

    @pytest.mark.parametrize("shape", [(5,), (3, 5)])
    def test_properties(self, shape):
        """Test statistical properties of Dirac distribution."""
        value = torch.randn(*shape)
        td_dirac = TensorDirac(value=value)

        # Mean should equal the value
        assert torch.allclose(td_dirac.mean, value)

        # Mode should equal the value
        assert torch.allclose(td_dirac.mode, value)

        # Variance should be zero
        assert torch.allclose(td_dirac.variance, torch.zeros_like(value))

        # Standard deviation should be zero
        assert torch.allclose(td_dirac.stddev, torch.zeros_like(value))

        # Entropy should be zero
        assert torch.allclose(td_dirac.entropy(), torch.zeros_like(value))

    @pytest.mark.parametrize("shape", [(5,), (3, 5)])
    def test_cdf(self, shape):
        """Test cumulative distribution function."""
        value = torch.randn(*shape)
        td_dirac = TensorDirac(value=value)

        # CDF should be 0 for values less than the point value
        less_value = value - 1.0
        cdf_less = td_dirac.cdf(less_value)
        assert torch.allclose(cdf_less, torch.zeros_like(value))

        # CDF should be 1 for values greater than or equal to the point value
        greater_value = value + 1.0
        cdf_greater = td_dirac.cdf(greater_value)
        assert torch.allclose(cdf_greater, torch.ones_like(value))

        # CDF at exact value should be 1
        cdf_exact = td_dirac.cdf(value)
        assert torch.allclose(cdf_exact, torch.ones_like(value))


class TestTensorDiracTensorOperations:
    """Test TensorContainer operations on TensorDirac."""

    def test_device_transfer(self):
        """Test moving distribution to different devices."""
        value = torch.randn(5, 3)
        td_dirac = TensorDirac(value=value)

        # CPU to CPU should work
        td_cpu = td_dirac.to("cpu")
        assert td_cpu._value.device == torch.device("cpu")
        assert torch.allclose(td_cpu._value, value)

        # Test that samples still work after device transfer
        sample = td_cpu.sample()
        assert torch.allclose(sample, value)

    def test_unflatten_reconstruction(self):
        """Test distribution reconstruction from flattened attributes."""
        value = torch.randn(5, 3)
        td_dirac = TensorDirac(value=value, validate_args=True)

        # Simulate flattening and unflattening
        attributes = {
            "_value": td_dirac._value,
            "_validate_args": td_dirac._validate_args,
        }

        reconstructed = TensorDirac._unflatten_distribution(attributes)
        assert torch.allclose(reconstructed._value, value)
        assert reconstructed._validate_args is True

    def test_value_property(self):
        """Test the value property accessor."""
        value = torch.randn(5, 3)
        td_dirac = TensorDirac(value=value)

        assert torch.allclose(td_dirac.value, value)
        assert td_dirac.value is td_dirac._value


class TestDiracDistributionTolerance:
    """Test tolerance (atol, rtol) parameters for DiracDistribution."""

    def test_log_prob_with_default_tol(self):
        """Test log_prob with default tolerance (exact equality)."""
        loc = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        x = torch.tensor([[0.0, 1.1], [1.99, 3.0]])
        dist = DiracDistribution(loc)
        expected_log_prob = torch.tensor([[0.0, float("-inf")], [float("-inf"), 0.0]])
        log_prob = dist.log_prob(x)
        assert torch.equal(log_prob, expected_log_prob)

    def test_log_prob_with_nonzero_atol(self):
        """Test log_prob with non-zero absolute tolerance."""
        loc = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        x = torch.tensor([[0.0, 1.1], [1.99, 3.0]])
        dist = DiracDistribution(loc, atol=0.05)
        expected_log_prob = torch.tensor([[0.0, float("-inf")], [0.0, 0.0]])
        log_prob = dist.log_prob(x)
        assert torch.equal(log_prob, expected_log_prob)

    def test_log_prob_with_nonzero_rtol(self):
        """Test log_prob with non-zero relative tolerance."""
        loc = torch.tensor([[0.0, 1.0], [100.0, 100.0]])
        x = torch.tensor([[0.0, 1.1], [100.1, 103.0]])
        dist = DiracDistribution(loc, rtol=0.01)
        expected_log_prob = torch.tensor([[0.0, float("-inf")], [0.0, float("-inf")]])
        log_prob = dist.log_prob(x)
        assert torch.equal(log_prob, expected_log_prob)

    def test_log_prob_with_atol_bcasts_loc(self):
        """Test log_prob with atol broadcasting over loc."""
        loc = torch.tensor([100.0, 200.0])
        atol = torch.tensor([[2.0], [4.0]])
        x = torch.tensor([[102.0, 201.9], [99.1, 205.0]])
        dist = DiracDistribution(loc, atol=atol)
        expected_log_prob = torch.tensor([[0.0, 0.0], [0.0, float("-inf")]])
        log_prob = dist.log_prob(x)
        assert torch.equal(log_prob, expected_log_prob)

    def test_log_prob_with_rtol_bcasts_loc(self):
        """Test log_prob with rtol broadcasting over loc."""
        loc = torch.tensor([100.0, 200.0])
        rtol = torch.tensor([[0.01], [0.02]])
        x = torch.tensor([[102.0, 201.9], [99.1, 205.0]])
        dist = DiracDistribution(loc, rtol=rtol)
        expected_log_prob = torch.tensor([[float("-inf"), 0.0], [0.0, float("-inf")]])
        log_prob = dist.log_prob(x)
        assert torch.equal(log_prob, expected_log_prob)

    def test_cdf_with_default_tol(self):
        """Test CDF with default tolerance."""
        loc = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        x = torch.tensor([[-1.0, -0.1], [-0.01, 1.000001]])
        dist = DiracDistribution(loc)
        expected_cdf = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
        cdf = dist.cdf(x)
        assert torch.equal(cdf, expected_cdf)

    def test_cdf_with_nonzero_atol(self):
        """Test CDF with non-zero absolute tolerance."""
        loc = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        x = torch.tensor([[-1.0, -0.1], [-0.01, 1.000001]])
        dist = DiracDistribution(loc, atol=0.05)
        expected_cdf = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        cdf = dist.cdf(x)
        assert torch.equal(cdf, expected_cdf)

    def test_cdf_with_nonzero_rtol(self):
        """Test CDF with non-zero relative tolerance."""
        loc = torch.tensor([[1.0, 1.0], [100.0, 100.0]])
        x = torch.tensor([[0.9, 1.0], [99.9, 97.0]])
        dist = DiracDistribution(loc, rtol=0.01)
        expected_cdf = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        cdf = dist.cdf(x)
        assert torch.equal(cdf, expected_cdf)

    def test_invalid_atol_raises(self):
        """Test that negative atol raises an error when validate_args=True."""
        loc = torch.randn(2, 3)
        with pytest.raises((ValueError, RuntimeError)):
            dist = DiracDistribution(loc, atol=-1.0, validate_args=True)
            dist.log_prob(torch.zeros_like(loc))

    def test_invalid_rtol_raises(self):
        """Test that negative rtol raises an error when validate_args=True."""
        loc = torch.randn(2, 3)
        with pytest.raises((ValueError, RuntimeError)):
            dist = DiracDistribution(loc, rtol=-1.0, validate_args=True)
            dist.log_prob(torch.zeros_like(loc))

    def test_invalid_icdf_value_raises(self):
        """Test that icdf raises ValueError for values outside [0, 1] when validate_args=True."""
        loc = torch.tensor([1.0, 2.0])
        dist = DiracDistribution(loc, validate_args=True)

        # Test with value < 0
        with pytest.raises(ValueError):
            dist.icdf(torch.tensor([-0.5, 0.5]))

        # Test with value > 1
        with pytest.raises(ValueError):
            dist.icdf(torch.tensor([0.5, 1.5]))

    def test_integer_loc_with_atol(self):
        """Test integer loc with absolute tolerance."""
        loc = torch.tensor([[0, 1], [2, 3]])
        x = torch.tensor([[0, 2], [4, 2]])
        dist = DiracDistribution(loc.float(), atol=1.0)
        expected_log_prob = torch.tensor([[0.0, 0.0], [float("-inf"), 0.0]])
        log_prob = dist.log_prob(x.float())
        assert torch.equal(log_prob, expected_log_prob)

    def test_tolerance_parameters_broadcast(self):
        """Test that atol and rtol broadcast correctly with value."""
        value = torch.tensor([1.0, 2.0, 3.0])
        atol = torch.tensor([[0.1], [0.2]])  # Shape (2, 1)
        dist = DiracDistribution(value, atol=atol)

        # Should broadcast to shape (2, 3)
        assert dist.value.shape == (2, 3)
        assert dist.atol.shape == (2, 3)

        # Test that tolerance is applied correctly
        x = torch.tensor([[1.05, 2.0, 3.0], [1.15, 2.0, 3.0]])
        log_prob = dist.log_prob(x)

        # First batch: atol=0.1, so 1.05 should match 1.0
        assert log_prob[0, 0] == 0.0
        # Second batch: atol=0.2, so 1.15 should match 1.0
        assert log_prob[1, 0] == 0.0

    def test_expand_preserves_tolerances(self):
        """Test that expand operation preserves atol and rtol."""
        value = torch.tensor([1.0, 2.0])
        atol = torch.tensor([0.1, 0.2])
        rtol = torch.tensor([0.01, 0.02])
        dist = DiracDistribution(value, atol=atol, rtol=rtol)

        # Expand to batch shape (3, 2)
        expanded = dist.expand((3, 2))
        assert expanded.value.shape == (3, 2)
        assert expanded.atol.shape == (3, 2)
        assert expanded.rtol.shape == (3, 2)

        # Check that values are preserved
        assert torch.all(expanded.value[0] == value)
        assert torch.all(expanded.atol[0] == atol)
        assert torch.all(expanded.rtol[0] == rtol)
