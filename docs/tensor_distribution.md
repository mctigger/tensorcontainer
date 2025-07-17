# TensorDistribution User Guide

`TensorDistribution` is a wrapper around `torch.distributions.Distribution` that makes it compatible with the `tensorcontainer` ecosystem. It allows PyTorch distributions to be treated like `TensorDataClass` or `TensorDict` objects, enabling them to be part of the same data structures and manipulated with the same batched operations (e.g., indexing, reshaping, device transfer). It solves the problem of integrating probabilistic models and sampling into structured tensor workflows.

---

## 1. Explanation (The "Big Picture")

This section explains the core ideas behind `TensorDistribution` to help you understand its purpose and design.

### Core Concepts

`TensorDistribution` is a specialized container that wraps `torch.distributions.Distribution` instances, allowing them to seamlessly integrate with `tensorcontainer`'s structured data management. It extends the capabilities of PyTorch distributions by enabling them to behave like `TensorDataClass` or `TensorDict` objects, supporting batching, indexing, and various tensor-like operations. This is particularly useful for probabilistic models where the output is a distribution over structured data, such as in variational autoencoders (VAEs) or reinforcement learning policies.

### When to use TensorDistribution vs. TensorDataClass/TensorDict?

| Feature | TensorDistribution | TensorDataClass/TensorDict |
| :---------------- | :-------------------------------------------- | :-------------------------------------------- |
| **Purpose** | Represents a batch of distributions | Represents concrete, realized tensor data |
| **Core Object** | Wraps `torch.distributions.Distribution` | Contains `torch.Tensor` objects |
| **Sampling** | Provides `.sample()` and `.rsample()` methods | Stores sampled or observed data |
| **Log Probability** | Implements `.log_prob()` | Not applicable (for data, not distributions) |
| **Use Case** | Probabilistic modeling, VAEs, RL policies | Structured data management, batch processing |

**When to choose `TensorDistribution`:**

*   When you need to represent a batch of distributions over structured data (e.g., a distribution over images and labels).
*   It is ideal for variational autoencoders (VAEs), reinforcement learning policies, or any model where the output is a probabilistic distribution.
*   When you want to apply tensor-like operations (indexing, reshaping, device transfer) directly to a collection of distributions.

**When to choose `TensorDataClass`/`TensorDict`:**

*   When you are working with concrete, realized tensor data, not distributions.
*   When you need a flexible or fixed-schema container for your actual tensor values.

---

## 2. Tutorial: Your First TensorDistribution

This tutorial will guide you through creating and using a basic `TensorDistribution`.

### Installation

`TensorDistribution` is part of the `tensorcontainer` library. Please refer to the main installation guide for instructions on how to install the package.

### A Hands-On Example

To define your first `TensorDistribution`, you typically instantiate it with a `torch.distributions.Distribution` object. `TensorDistribution` then provides a container-like interface for this distribution.

```python
import torch
from torch.distributions import Normal
from tensorcontainer.tensor_distribution import TensorNormal

# Create a batch of Normal distributions
# Here, we have 2 independent Normal distributions, each with a mean and std_dev
# The batch shape is torch.Size([2])
means = torch.tensor([0.0, 1.0])
std_devs = torch.tensor([1.0, 0.5])
normal_dist = Normal(loc=means, scale=std_devs)

# Wrap it with TensorDistribution
tdist = TensorNormal(normal_dist)

print(f"TensorDistribution: {tdist}")
# >>> TensorDistribution(Normal(loc: torch.Size([2]), scale: torch.Size([2])))

print(f"Batch shape: {tdist.shape}")
# >>> Batch shape: torch.Size([2])

# Access the underlying torch.distribution instance
print(f"Underlying distribution: {tdist.dist}")
# >>> Underlying distribution: Normal(loc: torch.Size([2]), scale: torch.Size([2]))

# Sample from the distribution
sample = tdist.sample()
print(f"Sampled value: {sample}")
# >>> Sampled value: tensor([-0.0470,  1.0800]) (values will vary)

print(f"Sampled value shape: {sample.shape}")
# >>> Sampled value shape: torch.Size([2])
```

This example demonstrates how `TensorDistribution` wraps a PyTorch distribution and provides a familiar interface for batch operations.

---

## 3. How-To Guides (Practical Recipes)

This section provides a collection of goal-oriented guides to solve specific problems.

### How to Sample from a TensorDistribution

`TensorDistribution` provides `.sample()` and `.rsample()` methods, which behave similarly to their `torch.distributions` counterparts but return a `torch.Tensor` containing the sampled values.

```python
import torch
from torch.distributions import Normal
from tensorcontainer.tensor_distribution import TensorNormal

# Create a batch of Normal distributions using TensorNormal
means = torch.tensor([0.0, 1.0])
std_devs = torch.tensor([1.0, 0.5])
tdist = TensorNormal(loc=means, scale=std_devs)

# Sample from the distribution
sample = tdist.sample()
print(f"Sampled value: {sample}")
# >>> Sampled value: tensor([-0.0470,  1.0800]) (values will vary)
print(f"Sampled value type: {type(sample)}")
# >>> Sampled value type: <class 'torch.Tensor'>

# Resample (reparameterized sample)
rsample = tdist.rsample()
print(f"Reparameterized Sample: {rsample}")
# >>> Reparameterized Sample: tensor([-0.0470,  1.0800]) (values will vary)
```

### How to Compute Log Probability

The `.log_prob()` method computes the log probability of a given sample under the distribution. It returns the result within the same container structure as the sample.

```python
import torch
from torch.distributions import Normal
from tensorcontainer.tensor_distribution import TensorNormal

# Create a batch of Normal distributions
means = torch.tensor([0.0, 1.0])
std_devs = torch.tensor([1.0, 0.5])
normal_dist = Normal(loc=means, scale=std_devs)
tdist = TensorNormal(normal_dist)

# Generate some samples (can be from tdist.sample() or external)
samples = torch.tensor([-0.1, 1.2])

# Compute log probability
log_probs = tdist.log_prob(samples)
print(f"Log Probabilities: {log_probs}")
# >>> Log Probabilities: tensor([-0.9234, -1.0800]) (values will vary)
```

### How to Apply Tensor-like Transformations

`TensorDistribution` instances support common tensor operations like `clone()`, `to(device)`, `cpu()`, and `cuda()`. These operations are applied uniformly to the underlying distribution's parameters, ensuring consistency.

```python
import torch
from torch.distributions import Normal
from tensorcontainer.tensor_distribution import TensorNormal

means = torch.tensor([0.0, 1.0])
std_devs = torch.tensor([1.0, 0.5])
normal_dist = Normal(loc=means, scale=std_devs)
tdist = TensorNormal(normal_dist)

# Clone the instance
cloned_tdist = tdist.clone()
print(f"Cloned TensorDistribution: {cloned_tdist}")

# Transfer to a different device (e.g., CUDA if available, otherwise CPU)
if torch.cuda.is_available():
    cuda_tdist = tdist.to("cuda")
    print(f"Device: {cuda_tdist.device}, Underlying dist device: {cuda_tdist.dist.loc.device}")
else:
    print("CUDA not available. Skipping .to('cuda') example.")

cpu_tdist = tdist.to("cpu")
print(f"Device: {cpu_tdist.device}, Underlying dist device: {cpu_tdist.dist.loc.device}")
```

### How to Index and Slice a TensorDistribution

You can index and slice `TensorDistribution` instances using standard Python indexing (`__getitem__`). This operation applies the indexing uniformly across the underlying distribution's parameters.

```python
import torch
from torch.distributions import Normal
from tensorcontainer.tensor_distribution import TensorNormal

means = torch.tensor([0.0, 1.0, 2.0])
std_devs = torch.tensor([1.0, 0.5, 0.2])
normal_dist = Normal(loc=means, scale=std_devs)
tdist = TensorNormal(normal_dist)

print(f"Original TensorDistribution: {tdist}")
print(f"Original batch shape: {tdist.shape}")

# Slicing the first element
sliced_tdist = tdist[0]
print(f"Sliced TensorDistribution: {sliced_tdist}")
print(f"Sliced batch shape: {sliced_tdist.shape}")

# Slicing a range
range_sliced_tdist = tdist[1:3]
print(f"Range Sliced TensorDistribution: {range_sliced_tdist}")
print(f"Range Sliced batch shape: {range_sliced_tdist.shape}")
```

### How to Manipulate Batch Shapes

`TensorDistribution` provides methods like `view()`, `reshape()`, `permute()`, `squeeze()`, `unsqueeze()`, and `expand()` to manipulate the batch shapes of the underlying distribution's parameters simultaneously.

```python
import torch
from torch.distributions import Normal
from tensorcontainer.tensor_distribution import TensorNormal

means = torch.randn(1, 4, 5)
std_devs = torch.ones(1, 4, 5)
normal_dist = Normal(loc=means, scale=std_devs)
tdist = TensorNormal(normal_dist)

print(f"Original batch_size: {tdist.shape}")

# Reshape the batch dimensions
reshaped_tdist = tdist.reshape(4, 5)
print(f"Reshaped batch_size: {reshaped_tdist.shape}")

# Squeeze a dimension
squeezed_tdist = tdist.squeeze(0)
print(f"Squeezed batch_size: {squeezed_tdist.shape}")
```

---

## 4. API Reference

This section provides a technical, exhaustive description of the `TensorDistribution` API.

### Initialization & Core Attributes

*   `__init__(self, dist: torch.distributions.Distribution)`: Constructor for `TensorDistribution`.
    *   `dist`: The `torch.distributions.Distribution` instance to wrap.
*   `dist`: Property returning the underlying `torch.distributions.Distribution` instance.
*   `shape`: Property returning the `torch.Size` of the batch dimensions of the wrapped distribution.
*   `device`: Property returning the `torch.device` where the underlying distribution's parameters reside.

### Sampling Methods

*   `sample(self, sample_shape=torch.Size())`: Generates a sample or a batch of samples from the distribution. Returns a `TensorDataClass`, `TensorDict`, or `torch.Tensor` depending on the distribution's event shape and the structure of its parameters.
*   `rsample(self, sample_shape=torch.Size())`: Generates a reparameterized sample or a batch of samples from the distribution. Returns a `TensorDataClass`, `TensorDict`, or `torch.Tensor`.

### Probability Methods

*   `log_prob(self, value)`: Returns the log probability density/mass function evaluated at `value`. The `value` should have a compatible structure (e.g., `TensorDataClass`, `TensorDict`, or `torch.Tensor`).

### Tensor-like Operations

*   `clone()`: Returns a deep copy of the `TensorDistribution` instance and its underlying distribution's parameters.
*   `to(device)`: Moves the underlying distribution's parameters to the specified device.
*   `cpu()`: Moves the underlying distribution's parameters to CPU memory.
*   `cuda()`: Moves the underlying distribution's parameters to CUDA memory (if available).
*   `detach()`: Returns a new `TensorDistribution` instance with the underlying distribution's parameters detached from the current computation graph.

### Shape Manipulation

*   `view(*shape)`: Returns a new `TensorDistribution` instance with the batch dimensions reshaped according to the provided `shape`.
*   `reshape(*shape)`: Similar to `view`, but can handle non-contiguous memory.
*   `permute(*dims)`: Permutes the dimensions of the underlying distribution's parameters.
*   `squeeze(*dims)`: Removes singleton dimensions from the batch shape.
*   `unsqueeze(*dims)`: Adds singleton dimensions to the batch shape.
*   `expand(*sizes)`: Expands the batch dimensions of the underlying distribution's parameters.

---

## 5. Limitations

*   **Distribution-Specific Behavior:** While `TensorDistribution` provides a unified interface, the specific behavior of sampling, log probability, and other methods ultimately depends on the wrapped `torch.distributions.Distribution` implementation.
*   **Parameter Manipulation:** Direct manipulation of the underlying distribution's parameters (e.g., `tdist.dist.loc = new_loc`) should be done with care, as it might bypass `TensorDistribution`'s consistency checks. Prefer using `TensorDistribution`'s methods for transformations.
*   **No Direct Instantiation of Distribution Parameters:** `TensorDistribution` does not directly expose the parameters of the wrapped distribution for direct assignment (e.g., `tdist.loc = ...`). Instead, you interact with the distribution as a whole or through its `dist` attribute.