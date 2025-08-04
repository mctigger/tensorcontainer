import pytest
import torch
from torch.distributions import (
    TransformedDistribution as TorchTransformedDistribution,
)
from torch.distributions.transforms import AffineTransform, ExpTransform, Transform

from tensorcontainer.tensor_distribution.normal import TensorNormal
from tensorcontainer.tensor_distribution.transformed_distribution import (
    TransformedDistribution,
)
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_property_values_match,
)


class TestTransformedDistributionInitialization:
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
        loc = torch.randn(loc_shape)
        scale = torch.rand(scale_shape).exp()
        base_dist = TensorNormal(loc=loc, scale=scale)
        transforms: list[Transform] = [ExpTransform()]
        td = TransformedDistribution(base_distribution=base_dist, transforms=transforms)
        assert td.batch_shape == expected_batch_shape
        assert td.dist().batch_shape == expected_batch_shape

    def test_dist_method(self):
        loc = torch.randn(1)
        scale = torch.rand(1).exp()
        base_dist = TensorNormal(loc=loc, scale=scale)
        transforms = [ExpTransform(), AffineTransform(loc=1.0, scale=2.0)]
        td = TransformedDistribution(base_distribution=base_dist, transforms=transforms)

        torch_td = td.dist()
        assert isinstance(torch_td, TorchTransformedDistribution)
        assert isinstance(torch_td.base_dist, type(base_dist.dist()))
        assert torch.allclose(torch_td.base_dist.loc, base_dist.dist().loc)
        assert torch.allclose(torch_td.base_dist.scale, base_dist.dist().scale)
        assert torch_td.transforms == transforms


class TestTransformedDistributionOperations:
    @pytest.mark.parametrize("param_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, param_shape):
        loc = torch.randn(*param_shape)
        scale = torch.rand(*param_shape).exp()
        base_dist = TensorNormal(loc=loc, scale=scale)
        transforms: list[Transform] = [ExpTransform()]
        td = TransformedDistribution(base_distribution=base_dist, transforms=transforms)

        sample = td.sample()

        def sample_fn(td_instance):
            return td_instance.sample()

        def rsample_fn(td_instance):
            return td_instance.rsample()

        def log_prob_fn(td_instance, s):
            return td_instance.log_prob(s)

        run_and_compare_compiled(sample_fn, td, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td, sample, fullgraph=False)

    def test_sample_log_prob(self):
        loc = torch.randn(3, 5)
        scale = torch.rand(3, 5).exp()
        base_dist = TensorNormal(loc=loc, scale=scale)
        transforms: list[Transform] = [ExpTransform()]
        td = TransformedDistribution(base_distribution=base_dist, transforms=transforms)

        torch_td = TorchTransformedDistribution(base_dist.dist(), transforms)

        sample_shape = torch.Size([2, 1])
        sample = td.sample(sample_shape)
        torch_sample = torch_td.sample(sample_shape)
        assert torch_sample is not None
        assert sample.shape == torch_sample.shape
        assert torch.allclose(td.log_prob(sample), torch_td.log_prob(sample))

        rsample = td.rsample(sample_shape)
        torch_rsample = torch_td.rsample(sample_shape)
        assert torch_rsample is not None
        assert rsample.shape == torch_rsample.shape
        assert torch.allclose(td.log_prob(rsample), torch_td.log_prob(rsample))


class TestTransformedDistributionCopy:
    @pytest.fixture
    def base_distribution(self):
        """Create a base normal distribution for testing."""
        loc = torch.randn(3, 5)
        scale = torch.rand(3, 5).exp()
        return TensorNormal(loc=loc, scale=scale)

    @pytest.fixture
    def transforms(self):
        """Create a list of transforms for testing."""
        return [
            ExpTransform(),
            AffineTransform(loc=1.0, scale=2.0),
        ]

    @pytest.fixture
    def original_dist(self, base_distribution, transforms):
        """Create an original transformed distribution for testing."""
        return TransformedDistribution(
            base_distribution=base_distribution, transforms=transforms
        )

    def test_copy_creates_new_object(self, original_dist):
        """Test that copy creates a new object of the correct type."""
        copied_dist = original_dist.copy()

        # Check that the copy is a different object but same type
        assert copied_dist is not original_dist
        assert isinstance(copied_dist, TransformedDistribution)

    def test_copy_base_distribution_handling(self, original_dist):
        """Test that the base distribution is handled correctly in copy."""
        copied_dist = original_dist.copy()

        # Check that the base distribution is a different object
        assert copied_dist.base_distribution is not original_dist.base_distribution
        assert isinstance(copied_dist.base_distribution, TensorNormal)

        # Check that tensor parameters are the same objects (identity)
        original_base = original_dist.base_distribution
        copied_base = copied_dist.base_distribution

        assert original_base._loc is copied_base._loc
        assert original_base._scale is copied_base._scale

    def test_copy_transforms_handling(self, original_dist):
        """Test that transforms are handled correctly in copy."""
        copied_dist = original_dist.copy()

        # Check that the transforms are the same objects (they're not tensors)
        assert copied_dist.transforms is original_dist.transforms

    def test_copy_sampling_consistency(self, original_dist):
        """Test that copied distribution produces consistent sampling results."""
        copied_dist = original_dist.copy()
        sample_shape = torch.Size([2, 1])

        # Check that samples have the same shape
        original_sample = original_dist.sample(sample_shape)
        copied_sample = copied_dist.sample(sample_shape)
        assert original_sample.shape == copied_sample.shape

        # Check that log_prob values are consistent for the same sample
        torch.testing.assert_close(
            original_dist.log_prob(original_sample),
            copied_dist.log_prob(original_sample),
        )

    def test_copy_property_consistency(self, original_dist):
        """Test that copied distribution has the same properties."""
        copied_dist = original_dist.copy()

        # Check that the distributions have the same properties
        assert original_dist.batch_shape == copied_dist.batch_shape
        assert original_dist.device == copied_dist.device


class TestTransformedDistributionAPIMatch:
    def test_properties_match(self):
        loc = torch.randn(3, 5)
        scale = torch.rand(3, 5).exp()
        base_dist = TensorNormal(loc=loc, scale=scale)
        transforms: list[Transform] = [ExpTransform()]
        td = TransformedDistribution(base_distribution=base_dist, transforms=transforms)
        assert_property_values_match(td)
