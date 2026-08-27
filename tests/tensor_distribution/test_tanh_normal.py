import math

import torch
from torch.distributions import Normal, TanhTransform, TransformedDistribution

from src.tensorcontainer.tensor_distribution.tanh_normal import (
    ClampedTanhTransform,
    TensorTanhNormal,
)
from tests.compile_utils import run_and_compare_compiled


class TestTensorTanhNormal:
    def test_initialization_scalar(self):
        # Test with scalar loc and scale
        loc = torch.tensor(0.0)
        scale = torch.tensor(1.0)
        dist = TensorTanhNormal(loc, scale)
        assert dist.loc == loc
        assert dist.scale == scale
        assert dist.shape == torch.Size([])
        assert dist.device == loc.device

    def test_initialization_tensor(self):
        # Test with tensor loc and scale
        loc = torch.tensor([0.0, 1.0])
        scale = torch.tensor([1.0, 2.0])
        dist = TensorTanhNormal(loc, scale)
        assert torch.equal(dist.loc, loc)
        assert torch.equal(dist.scale, scale)
        assert dist.shape == loc.shape
        assert dist.device == loc.device

    def test_initialization_reinterpreted_batch_ndims(self):
        # Test with TensorIndependent wrapper for reinterpreted batch dimensions
        from src.tensorcontainer.tensor_distribution.independent import (
            TensorIndependent,
        )

        loc = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        base_dist = TensorTanhNormal(loc, scale)
        dist = TensorIndependent(base_dist, reinterpreted_batch_ndims=1)

        assert torch.equal(base_dist.loc, loc)
        assert torch.equal(base_dist.scale, scale)
        assert base_dist.shape == loc.shape
        assert base_dist.device == loc.device
        assert dist.reinterpreted_batch_ndims == 1

    def test_rsample(self):
        loc = torch.tensor([0.0, 1.0])
        scale = torch.tensor([1.0, 0.5])
        tensor_dist = TensorTanhNormal(loc, scale)

        sample_shape = torch.Size((100,))
        samples = tensor_dist.rsample(sample_shape)
        assert samples.shape == sample_shape + loc.shape

    def test_log_prob(self):
        loc = torch.tensor([0.0, 1.0])
        scale = torch.tensor([1.0, 0.5])
        tensor_dist = TensorTanhNormal(loc, scale)

        sample_shape = torch.Size((100,))
        samples = tensor_dist.rsample(sample_shape)

        # Check log_prob
        # Use the same samples for both log_prob calculations
        log_probs = tensor_dist.log_prob(samples)

        # The log_prob of the TensorTanhNormal should be equivalent to the log_prob
        # of the underlying TransformedDistribution.
        reference_dist = tensor_dist.dist()
        reference_log_probs = reference_dist.log_prob(samples)

        assert torch.allclose(log_probs, reference_log_probs, atol=1e-5)

    def test_compile_compatibility_rsample(self):
        loc = torch.tensor([0.0, 1.0])
        scale = torch.tensor([1.0, 0.5])
        dist = TensorTanhNormal(loc, scale, validate_args=False)

        # Test rsample
        run_and_compare_compiled(dist.rsample, torch.Size((5,)))

    def test_compile_compatibility_log_prob(self):
        loc = torch.tensor([0.0, 1.0])
        scale = torch.tensor([1.0, 0.5])
        dist = TensorTanhNormal(loc, scale, validate_args=False)

        # Test log_prob
        value = dist.rsample(torch.Size((1,)))
        run_and_compare_compiled(dist.log_prob, value)

    def test_property_types(self):
        """Test that properties exist and return tensors."""
        loc = torch.tensor([0.0, 1.0])
        scale = torch.tensor([1.0, 0.5])
        dist = TensorTanhNormal(loc, scale)

        # Test that properties exist and return tensors
        assert isinstance(dist.loc, torch.Tensor)
        assert isinstance(dist.scale, torch.Tensor)
        assert isinstance(dist.mean, torch.Tensor)
        assert isinstance(dist.variance, torch.Tensor)
        assert isinstance(dist.stddev, torch.Tensor)

    def test_basic_property_values(self):
        """Test basic property values match input parameters."""
        loc = torch.tensor([0.0, 1.0])
        scale = torch.tensor([1.0, 0.5])
        dist = TensorTanhNormal(loc, scale)

        # Test basic property values
        assert torch.equal(dist.loc, loc)
        assert torch.equal(dist.scale, scale)

    def test_mean_range_validation(self):
        """Test that mean is in the valid range (-1, 1) for tanh distribution."""
        loc = torch.tensor([0.0, 1.0])
        scale = torch.tensor([1.0, 0.5])
        dist = TensorTanhNormal(loc, scale)

        # Test that mean is in the valid range (-1, 1) for tanh distribution
        assert torch.all(dist.mean >= -1.0) and torch.all(dist.mean <= 1.0)

    def test_variance_and_stddev_properties(self):
        """Test variance and stddev properties."""
        loc = torch.tensor([0.0, 1.0])
        scale = torch.tensor([1.0, 0.5])
        dist = TensorTanhNormal(loc, scale)

        # Test that variance and stddev are positive
        assert torch.all(dist.variance >= 0.0)
        assert torch.all(dist.stddev >= 0.0)

        # Test that stddev is square root of variance
        assert torch.allclose(dist.stddev, torch.sqrt(dist.variance), atol=1e-6)

    def test_property_shapes_scalar(self):
        """Test that properties have correct shapes for scalar case."""
        # Test scalar case
        loc_scalar = torch.tensor(0.0)
        scale_scalar = torch.tensor(1.0)
        dist_scalar = TensorTanhNormal(loc_scalar, scale_scalar)

        assert dist_scalar.mean.shape == torch.Size([])
        assert dist_scalar.variance.shape == torch.Size([])
        assert dist_scalar.stddev.shape == torch.Size([])

    def test_property_shapes_tensor(self):
        """Test that properties have correct shapes for tensor case."""
        # Test tensor case
        loc_tensor = torch.tensor([0.0, 1.0, -0.5])
        scale_tensor = torch.tensor([1.0, 0.5, 2.0])
        dist_tensor = TensorTanhNormal(loc_tensor, scale_tensor)

        expected_shape = loc_tensor.shape
        assert dist_tensor.mean.shape == expected_shape
        assert dist_tensor.variance.shape == expected_shape
        assert dist_tensor.stddev.shape == expected_shape


class TestClampedTanhTransform:
    def test_inverse_is_finite_on_the_bounds(self):
        transform = ClampedTanhTransform()
        y = torch.tensor([1.0, -1.0, 0.0])
        x = transform.inv(y)
        assert torch.isfinite(x).all()
        torch.testing.assert_close(torch.tanh(x), y)

    def test_inverse_is_finite_on_the_bounds_in_half_precision(self):
        transform = ClampedTanhTransform()
        for dtype in (torch.float16, torch.bfloat16, torch.float64):
            x = transform.inv(torch.tensor([1.0, -1.0], dtype=dtype))
            assert torch.isfinite(x).all(), dtype

    def test_log_abs_det_jacobian_matches_torch_tanh_transform(self):
        transform = ClampedTanhTransform()
        x = torch.linspace(-12.0, 12.0, 49)
        y = torch.tanh(x)
        torch.testing.assert_close(
            transform.log_abs_det_jacobian(x, y),
            TanhTransform().log_abs_det_jacobian(x, y),
        )

    def test_log_abs_det_jacobian_is_finite_and_decreasing_past_saturation(self):
        # y = tanh(x) is exactly 1.0 in float32 for these x; the log-det must keep
        # tracking log(4) - 2x instead of flattening out at a floor.
        transform = ClampedTanhTransform()
        x = torch.tensor([10.0, 20.0, 40.0])
        y = torch.tanh(x)
        assert (y == 1.0).all()
        log_det = transform.log_abs_det_jacobian(x, y)
        assert torch.isfinite(log_det).all()
        torch.testing.assert_close(log_det, math.log(4.0) - 2.0 * x)


class TestTensorTanhNormalSaturation:
    """A tanh-squashed Normal with a scale of a few units saturates float32 tanh."""

    def test_log_prob_matches_torch_off_saturation(self):
        loc = torch.tensor([0.0, 1.0, -2.0])
        scale = torch.tensor([1.0, 0.5, 2.0])
        dist = TensorTanhNormal(loc, scale)
        reference = TransformedDistribution(Normal(loc, scale), [TanhTransform()])

        value = torch.tanh(torch.linspace(-4.0, 4.0, 33)).unsqueeze(-1).expand(-1, 3)
        torch.testing.assert_close(
            dist.log_prob(value), reference.log_prob(value), rtol=1e-5, atol=1e-5
        )

    def test_log_prob_is_finite_on_saturated_values(self):
        loc = torch.zeros(4)
        scale = torch.full((4,), 5.0)
        dist = TensorTanhNormal(loc, scale)

        value = torch.tensor([1.0, -1.0, 1.0, -1.0])
        assert torch.isfinite(dist.log_prob(value)).all()

    def test_rsample_log_prob_round_trip_is_finite(self):
        torch.manual_seed(0)
        loc = torch.zeros(256)
        scale = torch.full((256,), 5.0)
        dist = TensorTanhNormal(loc, scale)

        samples = dist.rsample(torch.Size((100,)))
        # The premise of the test: this scale really does saturate float32.
        assert (samples.abs() == 1.0).any()
        assert torch.isfinite(dist.log_prob(samples)).all()

    def test_entropy_is_finite_with_finite_gradients(self):
        torch.manual_seed(0)
        loc = torch.zeros(64, requires_grad=True)
        scale = torch.full((64,), 5.0, requires_grad=True)
        dist = TensorTanhNormal(loc, scale)

        entropy = dist.entropy()
        assert torch.isfinite(entropy).all()
        # The support is (-1, 1), so no estimate should exceed log 2 by more than
        # Monte-Carlo noise.
        assert (entropy < math.log(2.0) + 0.5).all()

        entropy.sum().backward()
        assert torch.isfinite(loc.grad).all()
        assert torch.isfinite(scale.grad).all()

    def test_mode_and_moments_are_finite_at_large_scale(self):
        loc = torch.zeros(16)
        scale = torch.full((16,), 5.0)
        dist = TensorTanhNormal(loc, scale)

        for value in (dist.mode, dist.mean, dist.stddev):
            assert torch.isfinite(value).all()
            assert (value.abs() <= 1.0).all()
