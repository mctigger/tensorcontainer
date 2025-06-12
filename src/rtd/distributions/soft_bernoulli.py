import torch
from torch.distributions import Bernoulli


class SoftBernoulli(Bernoulli):
    def log_prob(self, value):
        # Compute log probabilities
        log_p = super().log_prob(torch.ones_like(value))  # log(p)
        log_1_p = super().log_prob(torch.zeros_like(value))  # log(1 - p)

        # Compute soft BCE using the original formula
        log_probs = value * log_p + (1 - value) * log_1_p

        return log_probs
