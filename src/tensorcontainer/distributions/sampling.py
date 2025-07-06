import torch
from torch.distributions import Distribution


class SamplingDistribution(Distribution):
    def __init__(self, base_distribution: Distribution, n=100):
        self.base_dist = base_distribution
        self.n = n

    def __getattr__(self, name):
        return getattr(self.base_dist, name)

    @property
    def has_rsample(self):
        return self.base_dist.has_rsample

    def rsample(self, sample_shape=torch.Size()):
        return self.base_dist.rsample(sample_shape)

    def sample(self, sample_shape=torch.Size()):
        return self.base_dist.sample(sample_shape)

    @property
    def mean(self):
        return self.base_dist.rsample((self.n,)).mean(0)

    @property
    def stddev(self):
        return self.base_dist.rsample((self.n,)).std(0)

    @property
    def variance(self):
        return self.base_dist.rsample((self.n,)).var(0)

    @property
    def mode(self):
        samples = self.base_dist.sample((self.n,))
        log_probs = self.base_dist.log_prob(samples).view(self.n, -1)
        index = torch.argmax(log_probs, dim=0)

        selected = torch.gather(samples.view(self.n, -1), 0, index.unsqueeze(0))
        return selected

    def entropy(self):
        samples = self.base_dist.rsample((self.n,))
        logprob = self.base_dist.log_prob(samples)
        return -logprob.mean(0)

    def log_prob(self, value):
        return self.base_dist.log_prob(value)
