import torch
from tensorcontainer.tensor_distribution import (
    TensorNormal,
    TensorBernoulli,
)
from tensorcontainer.tensor_distribution.base import TensorDistribution


def chain(distribution: TensorDistribution):
    # Event dimensions must stay constant. Ignore them here.
    distribution = distribution.view(2, 3, 4)
    distribution = distribution.permute(1, 0, 2)
    # You can call .detach() similar to Tensor.detach()
    distribution = distribution.detach()

    return distribution


# Create a TensorNormal with one event dimension
loc = torch.randn(2 * 3 * 4)
scale = torch.abs(torch.randn(2 * 3 * 4))
normal = TensorNormal(loc=loc, scale=scale)

# Execute the chain for TensorNormal
chain(normal)

# Execute the chain for TensorBernoulli
bernoulli = TensorBernoulli(logits=torch.randn(2, 3, 4))
chain(bernoulli)

# Works perfectly fine!
