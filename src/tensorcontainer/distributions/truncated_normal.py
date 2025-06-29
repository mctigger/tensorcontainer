from __future__ import annotations

import torch
import torch.distributions


class TruncatedNormal(torch.distributions.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def rsample(self, sample_shape=torch.Size()):
        event = super().rsample(sample_shape)

        event_clamp = torch.clamp(event, self.low + self.eps, self.high - self.eps)

        event = event - event.detach() + event_clamp.detach()

        return event

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)
