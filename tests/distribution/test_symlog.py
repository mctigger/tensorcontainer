import pytest
import torch
from torch.distributions import constraints

from tensorcontainer.distributions.symlog import (
    SymLogDistribution,
    SymexpTransform,
    symexp,
    symlog,
)


class TestSymlogTransformations:
    """Test the basic symlog/symexp transformation functions."""

    def test_symlog_symexp_inverse_relationship(self):
        """Test that symlog and symexp are inverse functions."""
        x = torch.tensor([-10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 10.0])

        # Test symlog(symexp(x)) = x
        assert torch.allclose(symlog(symexp(x)), x, atol=1e-6)

        # Test symexp(symlog(x)) = x
        assert torch.allclose(symexp(symlog(x)), x, atol=1e-6)

    def test_symlog_properties(self):
        """Test basic properties of symlog function."""
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = symlog(x)

        # symlog should preserve sign
        assert torch.all(torch.sign(result) == torch.sign(x))

        # symlog(0) should be 0
        assert symlog(torch.tensor(0.0)) == 0.0

    def test_symexp_properties(self):
        """Test basic properties of symexp function."""
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = symexp(x)

        # symexp should preserve sign
        assert torch.all(torch.sign(result) == torch.sign(x))

        # symexp(0) should be 0
        assert symexp(torch.tensor(0.0)) == 0.0

    def test_symlog_edge_cases(self):
        """Test symlog function with edge cases."""
        # Test with very large values
        large_pos = torch.tensor([1e10])
        large_neg = torch.tensor([-1e10])

        # Should compress large values
        assert torch.allclose(
            symlog(large_pos), torch.log(torch.tensor(1 + 1e10)), atol=1e-6
        )
        assert torch.allclose(
            symlog(large_neg), -torch.log(torch.tensor(1 + 1e10)), atol=1e-6
        )

        # Test with very small values
        small_pos = torch.tensor([1e-10])
        small_neg = torch.tensor([-1e-10])

        # Should approximately equal the input for small values
        assert torch.allclose(symlog(small_pos), small_pos, atol=1e-10)
        assert torch.allclose(symlog(small_neg), small_neg, atol=1e-10)

        # Test with inf and -inf
        inf_pos = torch.tensor([float("inf")])
        inf_neg = torch.tensor([float("-inf")])

        assert torch.isinf(symlog(inf_pos))
        assert torch.isinf(symlog(inf_neg))
        assert symlog(inf_pos) > 0
        assert symlog(inf_neg) < 0

    def test_symexp_edge_cases(self):
        """Test symexp function with edge cases."""
        # Test with very large values
        large_pos = torch.tensor([10.0])  # exp(10) is already very large
        large_neg = torch.tensor([-10.0])

        # Should expand large values
        assert torch.allclose(
            symexp(large_pos), torch.exp(torch.tensor(10.0)) - 1, atol=1e-6
        )
        assert torch.allclose(
            symexp(large_neg), -(torch.exp(torch.tensor(10.0)) - 1), atol=1e-6
        )

        # Test with very small values
        small_pos = torch.tensor([1e-10])
        small_neg = torch.tensor([-1e-10])

        # Should approximately equal the input for small values
        assert torch.allclose(symexp(small_pos), small_pos, atol=1e-10)
        assert torch.allclose(symexp(small_neg), small_neg, atol=1e-10)

        # Test with inf and -inf
        inf_pos = torch.tensor([float("inf")])
        inf_neg = torch.tensor([float("-inf")])

        assert torch.isinf(symexp(inf_pos))
        assert torch.isinf(symexp(inf_neg))
        assert symexp(inf_pos) > 0
        assert symexp(inf_neg) < 0


class TestSymexpTransform:
    """Test the SymexpTransform class."""

    def test_transform_properties(self):
        """Test basic properties of the transform."""
        transform = SymexpTransform()

        assert transform.bijective is True
        assert transform.domain == constraints.real
        assert transform.codomain == constraints.real
        assert transform.sign == 1

    def test_transform_call(self):
        """Test the forward transformation."""
        transform = SymexpTransform()
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

        expected = symexp(x)
        result = transform._call(x)

        assert torch.allclose(result, expected, atol=1e-6)

    def test_transform_inverse(self):
        """Test the inverse transformation."""
        transform = SymexpTransform()
        y = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

        expected = symlog(y)
        result = transform._inverse(y)

        assert torch.allclose(result, expected, atol=1e-6)

    def test_transform_bijective_property(self):
        """Test that the transform is bijective."""
        transform = SymexpTransform()
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

        # Test that forward and inverse are inverses
        y = transform._call(x)
        x_reconstructed = transform._inverse(y)

        assert torch.allclose(x_reconstructed, x, atol=1e-6)

    def test_log_abs_det_jacobian(self):
        """Test the log absolute determinant of the Jacobian."""
        transform = SymexpTransform()
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = transform._call(x)

        jacobian = transform.log_abs_det_jacobian(x, y)
        expected = torch.abs(x)

        assert torch.allclose(jacobian, expected, atol=1e-6)


class TestSymLogDistribution:
    """Test the new SymLogDistribution implementation."""

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
        return SymLogDistribution(sample_params["loc"], sample_params["scale"])

    def test_initialization(self, sample_params):
        """Test basic initialization and properties."""
        dist = SymLogDistribution(sample_params["loc"], sample_params["scale"])

        assert dist.batch_shape == sample_params["loc"].shape
        assert dist.event_shape == torch.Size()
        assert torch.allclose(dist.loc, sample_params["loc"])
        assert torch.allclose(dist.scale, sample_params["scale"])

    def test_initialization_with_validation(self, sample_params):
        """Test initialization with argument validation."""
        # Test with validate_args=True
        dist = SymLogDistribution(
            sample_params["loc"], sample_params["scale"], validate_args=True
        )
        assert dist.batch_shape == sample_params["loc"].shape
        assert dist.event_shape == torch.Size()

    def test_initialization_with_invalid_scale(self):
        """Test initialization with invalid scale values."""
        loc = torch.tensor([1.0, -2.0, 0.0])
        invalid_scale = torch.tensor([0.5, -1.0, 2.0])  # Negative scale

        with pytest.raises(
            ValueError,
            match="Expected parameter scale.*to satisfy the constraint GreaterThan",
        ):
            SymLogDistribution(loc, invalid_scale, validate_args=True)

    def test_mode_property(self, sample_params):
        """Test that mode is correctly computed as symexp(loc)."""
        dist = SymLogDistribution(sample_params["loc"], sample_params["scale"])
        expected_mode = symexp(sample_params["loc"])

        assert torch.allclose(dist.mode, expected_mode)
        assert dist.mode.shape == sample_params["loc"].shape

    def test_mean_property(self, sample_params):
        """Test that mean is approximated by mode."""
        dist = SymLogDistribution(sample_params["loc"], sample_params["scale"])

        # Mean should equal mode (approximation)
        mean = dist.mean
        mode = dist.mode

        assert torch.allclose(mean, mode)
        assert mean.shape == sample_params["loc"].shape
        assert torch.all(torch.isfinite(mean))

    def test_sampling(self, sample_params):
        """Test sampling functionality."""
        dist = SymLogDistribution(sample_params["loc"], sample_params["scale"])

        # Test default sampling
        sample = dist.sample()
        assert sample.shape == dist.batch_shape
        assert torch.all(torch.isfinite(sample))

        # Test sampling with specific shape
        sample_shape = torch.Size([10, 2])
        samples = dist.sample(sample_shape)
        assert samples.shape == sample_shape + dist.batch_shape
        assert torch.all(torch.isfinite(samples))

    def test_sampling_with_seed(self, sample_params):
        """Test sampling with reproducible results."""
        dist = SymLogDistribution(sample_params["loc"], sample_params["scale"])

        # Set seed for reproducibility
        torch.manual_seed(42)
        samples1 = dist.sample((100,))

        torch.manual_seed(42)
        samples2 = dist.sample((100,))

        assert torch.allclose(samples1, samples2)

    def test_log_prob(self, sample_params):
        """Test log probability computation."""
        dist = SymLogDistribution(sample_params["loc"], sample_params["scale"])

        # Test log_prob at mode (should be high probability)
        mode_log_prob = dist.log_prob(dist.mode)
        assert mode_log_prob.shape == dist.batch_shape
        assert torch.all(torch.isfinite(mode_log_prob))

        # Test log_prob for random values
        test_values = torch.tensor([0.0, 1.0, -1.0])
        log_probs = dist.log_prob(test_values)
        assert log_probs.shape == test_values.shape
        assert torch.all(torch.isfinite(log_probs))

    def test_log_prob_batch_computation(self, sample_params):
        """Test log probability computation with batched inputs."""
        dist = SymLogDistribution(sample_params["loc"], sample_params["scale"])

        # Test with batched values
        batch_values = torch.tensor([[0.0, 1.0, -1.0], [2.0, -2.0, 0.5]])
        log_probs = dist.log_prob(batch_values)

        # Should broadcast correctly
        expected_shape = torch.Size([2, 3])
        assert log_probs.shape == expected_shape
        assert torch.all(torch.isfinite(log_probs))

    def test_cdf_and_icdf(self, sample_params):
        """Test CDF and inverse CDF functionality."""
        dist = SymLogDistribution(sample_params["loc"], sample_params["scale"])

        # Test CDF
        test_values = torch.tensor([0.0, 1.0, -1.0])
        cdf_values = dist.cdf(test_values)

        assert cdf_values.shape == test_values.shape
        assert torch.all((cdf_values >= 0) & (cdf_values <= 1))
        assert torch.all(torch.isfinite(cdf_values))

        # Test ICDF
        prob_values = torch.tensor([0.1, 0.5, 0.9])
        icdf_values = dist.icdf(prob_values)

        assert icdf_values.shape == prob_values.shape
        assert torch.all(torch.isfinite(icdf_values))

        # Test CDF/ICDF inverse relationship
        reconstructed_probs = dist.cdf(icdf_values)
        assert torch.allclose(reconstructed_probs, prob_values, atol=1e-5)

        # Test monotonicity of CDF for each batch dimension separately
        for i in range(len(sample_params["loc"])):
            single_dist = SymLogDistribution(
                sample_params["loc"][i : i + 1], sample_params["scale"][i : i + 1]
            )
            sorted_values = torch.sort(test_values)[0]
            sorted_cdf = single_dist.cdf(sorted_values)
            assert torch.all(sorted_cdf[:-1] <= sorted_cdf[1:])

    def test_expand(self, sample_params):
        """Test the expand method."""
        dist = SymLogDistribution(sample_params["loc"], sample_params["scale"])

        # Test expanding to a larger batch shape
        new_batch_shape = torch.Size([2, 3])
        expanded_dist = dist.expand(new_batch_shape)

        assert expanded_dist.batch_shape == new_batch_shape
        assert expanded_dist.event_shape == dist.event_shape
        assert expanded_dist.loc.shape == new_batch_shape
        assert expanded_dist.scale.shape == new_batch_shape

        # Test that expanded distribution still works
        samples = expanded_dist.sample()
        assert samples.shape == new_batch_shape
        assert torch.all(torch.isfinite(samples))

    def test_expand_with_instance(self, sample_params):
        """Test the expand method with a provided instance."""
        dist = SymLogDistribution(sample_params["loc"], sample_params["scale"])

        # Create a new instance to expand into
        new_instance = SymLogDistribution(torch.zeros(1), torch.ones(1))

        new_batch_shape = torch.Size([2, 3])
        expanded_dist = dist.expand(new_batch_shape, new_instance)

        assert expanded_dist is new_instance
        assert expanded_dist.batch_shape == new_batch_shape
        assert expanded_dist.loc.shape == new_batch_shape
        assert expanded_dist.scale.shape == new_batch_shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_support(self, sample_params):
        """Test that the distribution works on GPU."""
        # Move parameters to GPU
        loc_gpu = sample_params["loc"].cuda()
        scale_gpu = sample_params["scale"].cuda()

        # Create distribution on GPU
        dist_gpu = SymLogDistribution(loc_gpu, scale_gpu)

        # Test sampling on GPU
        samples = dist_gpu.sample()
        assert samples.is_cuda
        assert torch.all(torch.isfinite(samples))

        # Test log_prob on GPU
        log_probs = dist_gpu.log_prob(samples)
        assert log_probs.is_cuda
        assert torch.all(torch.isfinite(log_probs))

        # Test properties on GPU
        assert dist_gpu.mode.is_cuda
        assert dist_gpu.mean.is_cuda

    def test_different_dtypes(self, sample_params):
        """Test that the distribution works with different dtypes."""
        for dtype in [torch.float32, torch.float64]:
            loc = sample_params["loc"].to(dtype)
            scale = sample_params["scale"].to(dtype)

            dist = SymLogDistribution(loc, scale)

            # Test sampling
            samples = dist.sample()
            assert samples.dtype == dtype

            # Test log_prob
            log_probs = dist.log_prob(samples)
            assert log_probs.dtype == dtype

            # Test properties
            assert dist.mode.dtype == dtype
            assert dist.mean.dtype == dtype
