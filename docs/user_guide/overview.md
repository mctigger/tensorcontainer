# TensorContainer User Guide

*A comprehensive introduction to structured tensor management in PyTorch*

## Introduction

TensorContainer is a PyTorch library that transforms how you work with structured tensor data. Instead of manually managing collections of individual tensors, TensorContainer provides unified containers that behave like single tensors while organizing complex, heterogeneous data structures.

### Why TensorContainer?

Compare the traditional approach with TensorContainer:

**Before: Manual Tensor Management**
```python
# Traditional approach - error-prone and verbose
def process_batch(observations, actions, rewards):
    observations = observations.to('cuda')
    actions = actions.to('cuda') 
    rewards = rewards.to('cuda')
    
    observations = observations.view(batch_size, -1)
    actions = actions.view(batch_size, -1)
    rewards = rewards.view(batch_size, -1)
    
    return observations.detach(), actions.detach(), rewards.detach()
```

**After: TensorContainer**
```python
# TensorContainer approach - concise and safe
def process_batch(batch):
    return batch.to('cuda').view(batch_size, -1).detach()

# Single line handles all tensors consistently
```

**Key Benefits:**
- **Unified Operations**: Apply tensor operations like `.to()`, `.view()`, and `.detach()` to entire data structures
- **Batch Safety**: Automatic validation ensures consistent batch dimensions across all tensors
- **PyTorch Integration**: Seamless compatibility with `torch.compile`, PyTree operations, and existing workflows
- **Type Safety**: Static typing support with IDE autocomplete and error checking

## Core Concepts

Before diving into specific container types, let's understand the fundamental concepts that apply to all TensorContainer implementations.

### Batch vs Event Dimensions

TensorContainer enforces a clear separation between **batch dimensions** and **event dimensions**:

- **Batch Dimensions**: The leading dimensions (defined by `shape`) that must be consistent across all tensors in a container. These represent your batching structure (e.g., batch size, sequence length).

- **Event Dimensions**: The trailing dimensions beyond the batch shape that can vary between tensors. These represent the actual data structure (e.g., feature dimensions, action spaces).

```python
import torch
from tensorcontainer import TensorDict

# Container with batch shape (4, 3) - 4 samples, 3 time steps
data = TensorDict({
    'observations': torch.randn(4, 3, 128),    # Event dims: (128,)
    'actions': torch.randn(4, 3, 6),           # Event dims: (6,)
    'rewards': torch.randn(4, 3),              # Event dims: ()
}, shape=(4, 3), device='cpu')

print(f"Container batch shape: {data.shape}")  # (4, 3)
# All tensors share the same batch dimensions (4, 3)
# but have different event dimensions
```

### Shape Operations

All shape operations (like `view`, `reshape`, `permute`) only affect batch dimensions, preserving event dimensions:

```python
# Reshape batch dimensions from (4, 3) to (12,)
reshaped = data.reshape(12)
print(f"Reshaped batch shape: {reshaped.shape}")  # (12,)

# Original event dimensions are preserved:
# observations: (12, 128), actions: (12, 6), rewards: (12,)
```

### Device Management

Containers enforce device consistency across all tensors (unless `device=None`):

```python
# Move entire container to CUDA
cuda_data = data.to('cuda')
print(f"Device: {cuda_data.device}")  # cuda:0

# All tensors are automatically moved:
print(f"Observations device: {cuda_data['observations'].device}")  # cuda:0
```

For scenarios requiring mixed devices, use `device=None`:

```python
# Allow mixed devices when needed
mixed_device_data = TensorDict({
    'cpu_data': torch.randn(32, 64),               # On CPU
    'gpu_data': torch.randn(32, 32).cuda(),       # On GPU
}, shape=(32,), device=None)  # device=None allows mixed devices

print(f"CPU tensor device: {mixed_device_data['cpu_data'].device}")  # cpu
print(f"GPU tensor device: {mixed_device_data['gpu_data'].device}")  # cuda:0
```

### Indexing and Slicing

Indexing operates on batch dimensions only:

```python
# Index into the first batch dimension
first_sample = data[0]
print(f"First sample shape: {first_sample.shape}")  # (3,)

# Slice across time dimension
first_timestep = data[:, 0]
print(f"First timestep shape: {first_timestep.shape}")  # (4,)
```

### Nested Containers and Operation Propagation

TensorContainers can be nested within other TensorContainers, and all operations automatically propagate to child containers. Different container types are fully compatible:

```python
from tensorcontainer import TensorDict, TensorDataClass

# Define a simple TensorDataClass
class AgentState(TensorDataClass):
    position: torch.Tensor
    health: torch.Tensor

# Create nested structure mixing container types
agent = AgentState(
    position=torch.randn(4, 2),
    health=torch.randn(4, 1),
    shape=(4,), device='cpu'
)

data = TensorDict({
    'agent': agent,  # TensorDataClass nested in TensorDict
    'reward': torch.randn(4, 1)
}, shape=(4,), device='cpu')

# Operations propagate to all nested containers
cuda_data = data.to('cuda')
# Both TensorDict and TensorDataClass moved to CUDA

reshaped_data = data.reshape(2, 2)
# All containers reshaped while preserving structure

first_sample = data[0]
# Both containers indexed together
```

This demonstrates that TensorDict and TensorDataClass work seamlessly together, with operations propagating uniformly across all nested containers regardless of type.

### Stacking and Concatenation

TensorContainers support PyTorch's stacking and concatenation operations:

```python
# Stack containers along a new dimension
batch1 = RLBatch(observations=torch.randn(16, 128), actions=torch.randn(16, 4), shape=(16,))
batch2 = RLBatch(observations=torch.randn(16, 128), actions=torch.randn(16, 4), shape=(16,))

stacked = torch.stack([batch1, batch2])  # New batch dim: (2, 16, ...)
print(f"Stacked shape: {stacked.shape}")  # (2, 16)

# Concatenate along existing dimension
concatenated = torch.cat([batch1, batch2])  # Extended batch dim: (32, ...)
print(f"Concatenated shape: {concatenated.shape}")  # (32,)
```


## Container Types

Now let's explore the three main container types and their specific use cases.

## TensorDict: Dictionary-Style Containers

TensorDict provides a dictionary-like interface for dynamic collections of tensors with shared batch dimensions.

### Basic Usage

```python
import torch
from tensorcontainer import TensorDict

# Create a TensorDict
data = TensorDict({
    'states': torch.randn(32, 128),
    'actions': torch.randn(32, 4),
    'rewards': torch.randn(32, 1),
    'done': torch.randint(0, 2, (32, 1))
}, shape=(32,), device='cpu')

# Dictionary-style access
states = data['states']
data['next_states'] = torch.randn(32, 128)

# Check available keys
print(list(data.keys()))  # ['states', 'actions', 'rewards', 'done', 'next_states']
```

### Nested Structures

TensorDict supports nested dictionaries, automatically converting them to nested TensorDict instances:

```python
# Nested structure
nested_data = TensorDict({
    'agent': {
        'position': torch.randn(32, 3),
        'velocity': torch.randn(32, 3)
    },
    'environment': {
        'obstacles': torch.randn(32, 10, 2),
        'goals': torch.randn(32, 2)
    }
}, shape=(32,), device='cpu')

# Access nested values
position = nested_data['agent']['position']
print(f"Position shape: {position.shape}")  # torch.Size([32, 3])

# Flatten nested keys for easier processing
flat_data = nested_data.flatten_keys()
print(list(flat_data.keys()))  # ['agent.position', 'agent.velocity', 'environment.obstacles', 'environment.goals']
```

### When to Use TensorDict

- **Dynamic schemas**: When your data structure changes during runtime
- **Exploratory development**: For prototyping and experimentation
- **Dictionary-like semantics**: When you need flexible key-based access
- **Nested data**: For hierarchical data structures

## TensorDataClass: Type-Safe Containers

TensorDataClass provides a dataclass-based approach with static typing, automatic field generation, and IDE support.

### Basic Usage

```python
import torch
from tensorcontainer import TensorDataClass

class RLBatch(TensorDataClass):
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    done: torch.Tensor

# Create with full type safety
batch = RLBatch(
    observations=torch.randn(32, 128),
    actions=torch.randn(32, 4),
    rewards=torch.randn(32, 1),
    done=torch.randint(0, 2, (32, 1)),
    shape=(32,),
    device='cpu'
)

# Type-safe field access with IDE autocomplete
obs = batch.observations  # IDE knows this is torch.Tensor
batch.actions = torch.randn(32, 8)  # Type-checked assignment
```

### Advanced Field Definitions

```python
from dataclasses import field
from typing import Optional, Dict, List, Any

class FlexibleBatch(TensorDataClass):
    # Required tensor fields
    states: torch.Tensor
    actions: torch.Tensor
    
    # Optional fields
    next_states: Optional[torch.Tensor] = None
    
    # Non-tensor metadata
    episode_ids: List[int] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Default tensor with factory function
    rewards: torch.Tensor = field(
        default_factory=lambda: torch.zeros(1)
    )

batch = FlexibleBatch(
    states=torch.randn(16, 64),
    actions=torch.randn(16, 2),
    shape=(16,),
    device='cpu'
)
```

### Inheritance and Composition

```python
class BaseBatch(TensorDataClass):
    observations: torch.Tensor
    actions: torch.Tensor

class ExtendedBatch(BaseBatch):
    rewards: torch.Tensor
    values: torch.Tensor

# Fields are inherited and merged
extended = ExtendedBatch(
    observations=torch.randn(8, 128),
    actions=torch.randn(8, 4),
    rewards=torch.randn(8, 1),
    values=torch.randn(8, 1),
    shape=(8,),
    device='cpu'
)
```

### When to Use TensorDataClass

- **Static schemas**: When your data structure is well-defined and stable
- **Type safety**: For production code requiring static type checking
- **IDE support**: When you want autocomplete and refactoring support
- **Performance**: Optimized memory layout with `slots=True`

## TensorDistribution: Probabilistic Containers

TensorDistribution wraps PyTorch distributions with tensor-like operations while maintaining full compatibility with the `torch.distributions` API.

### Basic Distributions

```python
import torch
from tensorcontainer.tensor_distribution import TensorNormal, TensorIndependent

# Create a multivariate normal using TensorIndependent
base_dist = TensorNormal(
    loc=torch.zeros(16, 3),
    scale=torch.ones(16, 3),
    shape=(16,),
    device='cpu'
)

# Make last dimension independent (treat as multivariate)
action_dist = TensorIndependent(base_dist, 1)

# Standard distribution operations
actions = action_dist.sample()  # Shape: (16, 3)
log_probs = action_dist.log_prob(actions)  # Shape: (16,) - summed over last dim

# Tensor-like operations on distributions
gpu_dist = action_dist.to('cuda')
reshaped_dist = action_dist.reshape(4, 4)

# Detach for stop-gradient operations
# .detach() flows through TensorIndependent into the TensorNormal parameters
detached_dist = action_dist.detach()
```

### Distribution Operations

```python
# All the standard torch.distributions operations work
mean = normal_dist.mean
variance = normal_dist.variance
support = normal_dist.support

# Tensor-like operations
detached_dist = normal_dist.detach()
expanded_dist = normal_dist.expand(64, -1)

# Device transfers
cuda_dist = normal_dist.to('cuda')
```

### When to Use TensorDistribution

- **Policy networks**: For neural network outputs representing probability distributions
- **Probabilistic models**: When working with uncertainty and sampling
- **Reinforcement learning**: For action distributions and value function modeling
- **Tensor operations on distributions**: When you need to apply tensor transformations to entire distributions

## torch.compile Integration

All TensorContainer types work seamlessly with PyTorch's `torch.compile` for optimized execution:

```python
@torch.compile
def process_batch(batch):
    # All tensor operations compile efficiently
    return batch.to('cuda').reshape(-1).detach()

@torch.compile  
def policy_forward(action_dist):
    # Distribution operations are compile-safe
    actions = action_dist.sample()
    log_probs = action_dist.log_prob(actions)
    return actions, log_probs

# Efficient compiled execution works with all container types
compiled_batch = process_batch(rl_batch)
actions, log_probs = policy_forward(action_dist)
```

## Common Patterns and Best Practices

### Efficient Construction

For performance-critical code, use `unsafe_construction()` to skip validation:

```python
from tensorcontainer import TensorContainer

# In tight loops where shapes are guaranteed correct
with TensorContainer.unsafe_construction():
    batches = []
    for batch_data in data_loader:
        batch = RLBatch(
            observations=batch_data['obs'],
            actions=batch_data['actions'],
            rewards=batch_data['rewards'],
            done=batch_data['done'],
            shape=(batch_size,),
            device='cuda'
        )
        batches.append(batch)
```

