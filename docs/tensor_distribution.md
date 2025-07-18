# TensorDistribution User Guide

`TensorDistribution` is a wrapper for `torch.distributions.Distribution` that makes it compatible with the `tensorcontainer` ecosystem. It enables PyTorch distributions to be treated as `TensorDataClass` or `TensorDict` objects, allowing them to be seamlessly integrated into structured data pipelines and manipulated with batched, tensor-like operations (e.g., indexing, reshaping, device transfer).

This guide assumes you are an expert in `torch.distributions`. It focuses on what `TensorDistribution` adds, not on the basics of distributions themselves.

---

## 1. What `TensorDistribution` Adds to `torch.distributions`

This section details how `TensorDistribution` enhances `torch.distributions` for structured, tensor-based workflows.

### Core Enhancement: `tensorcontainer` Integration

The main advantage of `TensorDistribution` is that it fully integrates `torch.distributions.Distribution` instances into the `tensorcontainer` ecosystem. By wrapping a distribution, you grant it tensor-like properties, including:

-   **Batching:** Treat a collection of distributions as a single, batch-aware entity.
-   **Indexing and Slicing:** Use standard `__getitem__` syntax to select subsets of your batched distributions.
-   **Shape Manipulation:** Apply operations like `.view()`, `.reshape()`, `.squeeze()`, and `.expand()` directly to the distribution batch.
-   **Device Portability:** Move distributions between devices (`.to()`, `.cuda()`, `.cpu()`) just like a tensor.

This is especially useful in areas like VAEs or RL, where models frequently produce distributions for structured data. `TensorDistribution` ensures these outputs can be handled with the same tooling as regular tensor data.

### When to Wrap with `TensorDistribution`

-   **Choose `TensorDistribution`** when your model's output is a *distribution* over a batch of data, and you need to apply tensor-like operations to that batch of distributions.
-   **Use `TensorDataClass` or `TensorDict`** when you are working with *concrete, realized data* that has already been sampled or observed.

---

## 2. Tutorial: Your First TensorDistribution

This tutorial demonstrates how to wrap a `torch.distributions.Distribution` and access its tensor-like features.


### Simplified Tensor Operations

`TensorDistribution` allows common tensor operations to be applied directly to the distribution object, which can simplify code. This differs from how these operations are typically performed on a standard `torch.distributions.Distribution` instance.

Methods like `.detach()` or `.view()` can be called directly on the `TensorDistribution` object, and these operations are applied uniformly to all underlying parameter tensors (e.g., `loc`, `scale`). Crucially, to apply these operations, you don't need to know the specific type of `TensorDistribution` (e.g., `TensorNormal`, `TensorCategorical`).

Here is an example to showcase how using TensorDistribution leads to more readable and flexible code:

```python
import torch
from torch.distributions import kl_divergence
from tensorcontainer.tensor_distribution import (
    TensorBernoulli,
    TensorCategorical,
    TensorNormal,
    TensorDistribution,
)


def partially_detached_kl_divergence(p: TensorDistribution, q: TensorDistribution):
    """
    Compute KL divergence between p and a detached version of q.
    
    With TensorDistribution, we can simply call .detach() on any distribution
    without needing to know its specific type or parameter names.
    """
    return kl_divergence(p.dist(), q.detach().dist())


# Create different types of TensorDistributions with gradients
normal = TensorNormal(
    loc=torch.tensor([0.0, 1.0], requires_grad=True),
    scale=torch.tensor([1.0, 0.5], requires_grad=True),
)
categorical = TensorCategorical(
    logits=torch.tensor([[0.1, 0.9], [0.8, 0.2]], requires_grad=True)
)
bernoulli = TensorBernoulli(probs=torch.tensor([0.2, 0.8], requires_grad=True))

# The same function works for all distribution types
# No type-specific handling required!
kl_normal = partially_detached_kl_divergence(normal, normal)
kl_categorical = partially_detached_kl_divergence(categorical, categorical)
kl_bernoulli = partially_detached_kl_divergence(bernoulli, bernoulli)
```

### Comparison with Standard torch.distributions
In contrast, with a `torch.distributions.Distribution`, you would need to manually access each underlying parameter tensor (e.g., `loc`, `scale`), apply the operation to each one individually, and then create a new distribution instance with the modified parameters. This process can be complex, repetitive, and prone to errors, especially since parameter names vary between different distribution types (e.g., `Normal` has `loc` and `scale`, while `Categorical` has `probs` or `logits`). A function taking a standard `Distribution` object requires manual, type-specific handling of parameters.

The following example demonstrates the complexity of working with standard `torch.distributions.Distribution` objects:

```python
import torch
from torch.distributions import (
    Normal,
    Categorical,
    Bernoulli,
    Distribution,
    kl_divergence,
)


def partially_detached_kl_divergence(p: Distribution, q: Distribution):
    """
    Compute KL divergence between p and a detached version of q.
    
    With standard torch.distributions, we need type-specific handling
    because different distributions have different parameter names and structures.
    """
    # Create detached version of q based on its type
    if isinstance(q, Normal):
        detached_q = Normal(loc=q.loc.detach(), scale=q.scale.detach())
    elif isinstance(q, Categorical):
        detached_q = Categorical(logits=q.logits.detach())
    elif isinstance(q, Bernoulli):
        detached_q = Bernoulli(probs=q.probs.detach())
    else:
        raise RuntimeError(
            f"partially_detached_kl_divergence not implemented for distribution {type(q)}"
        )

    return kl_divergence(p, detached_q)


# Create different types of distributions with gradients
normal = Normal(
    loc=torch.tensor([0.0, 1.0], requires_grad=True),
    scale=torch.tensor([1.0, 0.5], requires_grad=True),
)
categorical = Categorical(
    logits=torch.tensor([[0.1, 0.9], [0.8, 0.2]], requires_grad=True)
)
bernoulli = Bernoulli(probs=torch.tensor([0.2, 0.8], requires_grad=True))

# Each distribution type requires the same function but with type-specific logic
kl_normal = partially_detached_kl_divergence(normal, normal)
kl_categorical = partially_detached_kl_divergence(categorical, categorical)
kl_bernoulli = partially_detached_kl_divergence(bernoulli, bernoulli)
```

---

## 3. How-To Guides (Practical Recipes)

This section provides goal-oriented examples of `TensorDistribution`'s added capabilities.

### How to Sample from a TensorDistribution

The `.sample()` and `.rsample()` methods mirror their `torch.distributions` counterparts but are called on the `TensorDistribution` wrapper. They return a `torch.Tensor` of sampled values.

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
# sample is a torch.Tensor
# Values will vary

# Resample (reparameterized sample)
rsample = tdist.rsample()
# rsample is a torch.Tensor
# Values will vary
```

### How to Compute Log Probability

The `.log_prob()` method works as expected, computing the log probability of a given sample.

```python
import torch
from torch.distributions import Normal
from tensorcontainer.tensor_distribution import TensorNormal

# Create a batch of Normal distributions
means = torch.tensor([0.0, 1.0])
std_devs = torch.tensor([1.0, 0.5])
tdist = TensorNormal(loc=means, scale=std_devs)

# Generate some samples (can be from tdist.sample() or external)
samples = torch.tensor([-0.1, 1.2])

# Compute log probability
log_probs = tdist.log_prob(samples)
# log_probs is a tensor with log probabilities
# Values will vary
```

### How to Apply Tensor-like Transformations

This is a key advantage of `TensorDistribution`. Operations like `clone()`, `to(device)`, `cpu()`, and `cuda()` are applied to all underlying parameters of the distribution simultaneously.

```python
import torch
from torch.distributions import Normal
from tensorcontainer.tensor_distribution import TensorNormal

means = torch.tensor([0.0, 1.0])
std_devs = torch.tensor([1.0, 0.5])
tdist = TensorNormal(loc=means, scale=std_devs)

# Clone the instance
cloned_tdist = tdist.clone()
# cloned_tdist is a new TensorNormal instance with copied parameters

# Transfer to a different device (e.g., CUDA if available, otherwise CPU)
if torch.cuda.is_available():
    cuda_tdist = tdist.to("cuda")
    # cuda_tdist.device is torch.device('cuda:0')
    # cuda_tdist.dist().loc.device is torch.device('cuda:0')
else:
    # CUDA not available. Skipping .to('cuda') example.
    pass

cpu_tdist = tdist.to("cpu")
# cpu_tdist.device is torch.device('cpu')
# cpu_tdist.dist().loc.device is torch.device('cpu')
```

### How to Index and Slice a TensorDistribution

Standard `__getitem__` indexing can be used to slice the batch dimension of your distribution. This operation is applied uniformly to the underlying parameters.

```python
import torch
from torch.distributions import Normal
from tensorcontainer.tensor_distribution import TensorNormal

means = torch.tensor([0.0, 1.0, 2.0])
std_devs = torch.tensor([1.0, 0.5, 0.2])
tdist = TensorNormal(loc=means, scale=std_devs)

# Original batch shape is torch.Size([3])

# Slicing the first element
sliced_tdist = tdist[0]
# sliced_tdist.shape is torch.Size([])

# Slicing a range
range_sliced_tdist = tdist[1:3]
# range_sliced_tdist.shape is torch.Size([2])
```

### How to Manipulate Batch Shapes

`TensorDistribution` provides tensor-like methods (`view()`, `reshape()`, `permute()`, `squeeze()`, `unsqueeze()`, `expand()`) to manipulate the batch shape of the underlying distribution's parameters.

```python
import torch
from torch.distributions import Normal
from tensorcontainer.tensor_distribution import TensorNormal

means = torch.randn(1, 4, 5)
std_devs = torch.ones(1, 4, 5)
tdist = TensorNormal(loc=means, scale=std_devs)

# Original batch_size: torch.Size([1, 4, 5])

# Reshape the batch dimensions
reshaped_tdist = tdist.reshape(4, 5)
# Reshaped batch_size: torch.Size([4, 5])

# Squeeze a dimension
squeezed_tdist = tdist.squeeze(0)
# Squeezed batch_size: torch.Size([4, 5])
```
---

## 4. API Reference

This section provides a technical description of the `TensorDistribution` API.

### Initialization & Core Attributes

*   `__init__(self, dist: torch.distributions.Distribution)`: Constructor. Wraps a `torch.distributions.Distribution` instance.
*   `dist`: The underlying `torch.distributions.Distribution` instance.
*   `shape`: The `torch.Size` of the batch dimensions of the wrapped distribution.
*   `device`: The `torch.device` where the distribution's parameters reside.

### Sampling Methods

*   `sample(self, sample_shape=torch.Size())`: Generates samples. Returns a `torch.Tensor`.
*   `rsample(self, sample_shape=torch.Size())`: Generates reparameterized samples. Returns a `torch.Tensor`.

### Probability Methods

*   `log_prob(self, value)`: Returns the log probability of `value`.

### Tensor-like Operations

*   `clone()`: Returns a deep copy.
*   `to(device)`: Moves parameters to the specified device.
*   `cpu()`: Moves parameters to CPU memory.
*   `cuda()`: Moves parameters to CUDA memory.
*   `detach()`: Detaches parameters from the computation graph.

### Shape Manipulation

*   `view(*shape)`: Reshapes batch dimensions.
*   `reshape(*shape)`: Reshapes batch dimensions (non-contiguous memory safe).
*   `permute(*dims)`: Permutes batch dimensions.
*   `squeeze(*dims)`: Removes singleton dimensions.
*   `unsqueeze(*dims)`: Adds singleton dimensions.
*   `expand(*sizes)`: Expands batch dimensions.

---

## 5. Limitations

*   **Implementation-Dependent:** Behavior is determined by the underlying `torch.distributions.Distribution` instance.
*   **Parameter Immutability:** `TensorDistribution` methods should be preferred for transformations over direct manipulation of distribution parameters (e.g., `tdist.dist.loc = new_loc`), which can cause inconsistencies.