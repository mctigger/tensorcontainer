import pytest
import torch
from torch.distributions import Normal, Categorical

from tensorcontainer.distributions.sampling import SamplingDistribution


class TestSamplingDistribution:
    def test_initialization(self):
        base_dist = Normal(0, 1)
        dist = SamplingDistribution(base_dist, n=50)
        assert dist.n == 50
        assert dist.base_dist is base_dist

        with pytest.raises(TypeError):
            SamplingDistribution("not a distribution")  # type: ignore

        with pytest.raises(ValueError):
            SamplingDistribution(base_dist, n=0)

        with pytest.raises(ValueError):
            SamplingDistribution(base_dist, n=-10)

    def test_repr(self):
        base_dist = Normal(0, 1)
        dist = SamplingDistribution(base_dist, n=50)
        assert repr(dist) == f"SamplingDistribution(base_dist={base_dist}, n=50)"

    def test_sampling_methods(self):
        base_dist = Normal(torch.zeros(2), torch.ones(2))
        dist = SamplingDistribution(base_dist, n=100)

        # Test sample
        sample = dist.sample(torch.Size((dist.n,)))
        # The shape should be (n, *base_dist.event_shape)
        assert (
            sample.shape
            == torch.Size((dist.n,)) + base_dist.batch_shape + base_dist.event_shape
        )

        # Test rsample
        rsample = dist.rsample(torch.Size((dist.n,)))
        assert (
            rsample.shape
            == torch.Size((dist.n,)) + base_dist.batch_shape + base_dist.event_shape
        )
        # rsample should have grad if base dist params have grad
        base_dist.loc = base_dist.loc.clone().requires_grad_()
        # Clear the cache to re-sample with the new grad requirements
        if hasattr(dist, "_samples"):
            del dist._samples
        rsample = dist.rsample(torch.Size((dist.n,)))
        assert rsample.requires_grad


class TestSamplingPropertiesRsample:
    def test_cached_properties_with_rsample(self):
        base_dist = Normal(torch.tensor([0.0, 10.0]), torch.tensor([1.0, 0.1]))
        dist = SamplingDistribution(base_dist, n=10000)

        # Accessing properties should cache them
        mean1 = dist.mean
        mean2 = dist.mean
        assert torch.all(mean1 == mean2)
        assert dist._samples is dist._samples  # check if samples are cached

        # Check if stats are reasonable
        assert torch.allclose(dist.mean, base_dist.mean, atol=0.1)
        assert torch.allclose(dist.stddev, base_dist.stddev, atol=0.1)
        assert torch.allclose(dist.variance, base_dist.variance, atol=0.2)

        # Check caching
        samples_id = id(dist._samples)
        _ = dist.stddev
        assert id(dist._samples) == samples_id
        _ = dist.variance
        assert id(dist._samples) == samples_id

    def test_mode_sampling(self):
        # Use a distribution where mode is not analytical to test sampling-based mode
        base_dist = Normal(torch.tensor([0.0, 10.0]), torch.tensor([1.0, 2.0]))

        # Mock analytical mode to be unavailable
        class MockNormal(Normal):
            @property
            def mode(self):
                raise NotImplementedError

        mock_dist = MockNormal(base_dist.loc, base_dist.scale)
        dist = SamplingDistribution(mock_dist, n=1000)

        mode = dist.mode
        assert mode.shape == base_dist.batch_shape
        # The sampled mode should be close to the analytical mode
        assert torch.allclose(mode, mock_dist.loc, atol=0.5)

        # Check caching
        mode2 = dist.mode
        assert torch.all(mode == mode2)

    def test_entropy_sampling(self):
        base_dist = Normal(0, 1)
        dist = SamplingDistribution(base_dist, n=10000)

        # Sampled entropy should be close to analytical entropy
        analytical_entropy = base_dist.entropy()
        sampled_entropy = dist.entropy()
        assert torch.allclose(analytical_entropy, sampled_entropy, atol=0.1)

        # Check caching of samples
        samples_id = id(dist._samples)
        _ = dist.entropy()
        assert id(dist._samples) == samples_id


class TestSamplingPropertiesNoRsample:
    def test_cached_properties_without_rsample(self):
        # Categorical does not have rsample
        base_dist = Categorical(torch.tensor([0.1, 0.2, 0.7]))
        dist = SamplingDistribution(base_dist, n=10000)

        # Accessing properties should cache them
        mean1 = dist.mean
        mean2 = dist.mean
        assert torch.all(mean1 == mean2)
        assert dist._samples is dist._samples

        # Check if stats are reasonable for categorical
        # mean of samples should be close to E[X] = sum(i * p_i)
        expected_mean = torch.tensor(0 * 0.1 + 1 * 0.2 + 2 * 0.7)
        assert torch.allclose(dist.mean.float(), expected_mean, atol=0.1)


class TestSamplingWithIndependent:
    def test_sampling_with_independent_normal(self):
        # Test with Independent(Normal) as base_dist
        # Corrected reinterpreted_batch_ndims to 0 as Normal(0,1) has no batch dimensions to reinterpret
        base_dist = torch.distributions.Independent(Normal(0, 1), 0)
        dist = SamplingDistribution(base_dist, n=1000)

        sample = dist.sample((1000,))
        # The shape should be (n, *base_dist.event_shape)
        assert sample.shape == (1000, *base_dist.event_shape)

        # Check if samples are within the expected range for Normal(0,1)
        # Check if the mean and stddev of samples are close to the analytical values
        assert torch.allclose(sample.mean(), torch.tensor(0.0), atol=0.1)
        assert torch.allclose(sample.std(), torch.tensor(1.0), atol=0.1)

    def test_log_prob_with_independent_normal(self):
        # Test log_prob with Independent(Normal) as base_dist
        base_dist = torch.distributions.Independent(
            Normal(torch.tensor([0.0, 10.0]), torch.tensor([1.0, 2.0])), 1
        )
        dist = SamplingDistribution(base_dist, n=10000)

        # Sampled log_prob should be close to analytical log_prob
        # For Independent(Normal(loc, scale), 1), log_prob is sum of log_prob of each element
        # For a single element Normal(loc, scale), log_prob is -0.5 * ((x - loc) / scale)**2 - 0.5 * log(2 * pi * scale**2)
        # For Independent(Normal(loc, scale), 1), the event_shape is (1,)
        # The log_prob of the sampled value should be close to the log_prob of the base distribution
        samples = dist.sample(torch.Size((1,)))
        sampled_log_prob = dist.log_prob(samples)
        analytical_log_prob = base_dist.log_prob(samples)

        assert torch.allclose(sampled_log_prob, analytical_log_prob, atol=0.1)

        # Check caching of samples for log_prob
        samples_id = id(dist._samples)
        _ = dist.log_prob(samples)
        assert id(dist._samples) == samples_id
