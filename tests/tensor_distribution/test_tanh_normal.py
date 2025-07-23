import torch

from src.tensorcontainer.tensor_distribution.tanh_normal import (
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
        dist = TensorTanhNormal(loc, scale)

        # Test rsample
        run_and_compare_compiled(dist.rsample, torch.Size((5,)))

    def test_compile_compatibility_log_prob(self):
        loc = torch.tensor([0.0, 1.0])
        scale = torch.tensor([1.0, 0.5])
        dist = TensorTanhNormal(loc, scale)

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
