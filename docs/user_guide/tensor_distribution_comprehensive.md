# TensorDistribution User Guide

**What you'll learn:** How TensorDistribution makes working with probability distributions as simple and powerful as working with tensors. By the end of this guide, you'll understand why TensorDistribution is essential for probabilistic modeling and how it seamlessly integrates with your existing PyTorch workflows.

**The key insight:** Instead of managing collections of distributions manually, TensorDistribution lets you treat entire batches of distributions like single tensor objects - applying operations like `.to()`, `.view()`, and `.expand()` to transform all distributions at once.

## The Core Insight: Distributions as Tensors

The fundamental problem with `torch.distributions` is that distributions don't behave like tensors. When you have a batch of distributions, you can't easily move them to different devices, reshape them, or apply other tensor operations without writing boilerplate code.

TensorDistribution solves this by making distributions behave exactly like tensors. Here's the difference:

**With torch.distributions (the hard way):**
```python
import torch
from torch.distributions import Normal

# Create batch of normal distributions
locs = torch.zeros(32, 4)
scales = torch.ones(32, 4)
normal_dists = Normal(loc=locs, scale=scales)

# Want to move to GPU? You have to reconstruct everything
locs_gpu = locs.to('cuda')
scales_gpu = scales.to('cuda')  
normal_dists_gpu = Normal(loc=locs_gpu, scale=scales_gpu)

# Want to reshape from (32, 4) to (8, 4, 4)? More reconstruction
locs_reshaped = locs.view(8, 4, 4)
scales_reshaped = scales.view(8, 4, 4)
normal_dists_reshaped = Normal(loc=locs_reshaped, scale=scales_reshaped)
```

**With TensorDistribution (the easy way):**
```python
from tensorcontainer.tensor_distribution import TensorNormal

# Create batch of normal distributions
normal_dist = TensorNormal(
    loc=torch.zeros(32, 4),
    scale=torch.ones(32, 4),
    shape=(32,),
    device='cpu'
)

# Move to GPU with a single operation
normal_dist_gpu = normal_dist.to('cuda')

# Reshape with a single operation  
normal_dist_reshaped = normal_dist.view(8, 4)
```

This is the power of TensorDistribution: **you work with batches of distributions as if they were single objects**, just like you work with tensors.

## Getting Started

Let's start with a simple example that demonstrates the core concept. Imagine you're building a policy network that outputs action distributions for a batch of states.

**Creating your first TensorDistribution:**

```python
import torch
from tensorcontainer.tensor_distribution import TensorNormal

# Your policy network outputs these parameters for 32 different states
action_means = torch.randn(32, 6)      # Mean actions for each state
action_stds = torch.ones(32, 6)        # Standard deviations

# Create a batch of distributions - one for each state
action_dist = TensorNormal(
    loc=action_means,
    scale=action_stds,
    shape=(32,),                        # Batch shape: 32 states
    device='cpu'
)
```

Now you have a single object representing 32 different normal distributions. Each distribution has 6-dimensional actions, but you work with all 32 distributions as one unit.

**Sampling actions for all states at once:**

```python
# Sample one action for each of the 32 states
actions = action_dist.sample()          # Shape: (32, 6)

# Sample multiple action candidates for each state
action_candidates = action_dist.sample((10,))  # Shape: (10, 32, 6)
```

The beauty is that `action_dist` behaves like a tensor - you call methods on it once, and the operation applies to all 32 distributions simultaneously.

## Why TensorDistribution? Key Advantages

### 1. Effortless Device Management

Moving distributions between devices is a common pain point. With `torch.distributions`, you have to manually track and move every parameter tensor. TensorDistribution handles this automatically.

**The problem with torch.distributions:**
```python
# You have distributions on CPU
loc_cpu = torch.zeros(1000, 10)
scale_cpu = torch.ones(1000, 10)
dist_cpu = Normal(loc_cpu, scale_cpu)

# Moving to GPU requires reconstructing everything manually
loc_gpu = loc_cpu.to('cuda')
scale_gpu = scale_cpu.to('cuda') 
dist_gpu = Normal(loc_gpu, scale_gpu)
```

**TensorDistribution solution:**
```python
# Create distribution on CPU
dist = TensorNormal(
    loc=torch.zeros(1000, 10),
    scale=torch.ones(1000, 10),
    shape=(1000,),
    device='cpu'
)

# Move to GPU in one operation
dist_gpu = dist.to('cuda')
# All parameters are automatically moved and the distribution is reconstructed
```

This becomes crucial when you have complex nested structures with dozens of distributions that need to move between devices during training.

### 2. Intuitive Shape Transformations  

Reshaping batches of distributions is essential for many ML workflows, but `torch.distributions` makes this tedious. TensorDistribution makes it as easy as reshaping tensors.

**Common scenario:** You have a sequence of distributions that you want to flatten for processing, then reshape back.

```python
# Start with sequence of distributions: (time_steps=20, batch_size=16)
sequence_dist = TensorNormal(
    loc=torch.randn(20, 16, 5),
    scale=torch.ones(20, 16, 5),
    shape=(20, 16),
    device='cpu'
)

# Flatten for batch processing
flat_dist = sequence_dist.view(-1)     # Shape: (320,)
assert flat_dist.shape == (320,)

# Process in smaller chunks
chunk_dist = flat_dist.view(64, 5)    # Shape: (64, 5)
assert chunk_dist.shape == (64, 5)

# Sample from reshaped distributions
samples = chunk_dist.sample()          # Shape: (64, 5, 5)
```

With `torch.distributions`, each of these reshaping operations would require manually reshaping parameters and reconstructing distribution objects.

### 3. Powerful Indexing and Slicing

When working with batches of distributions, you often need to select specific distributions or apply operations to subsets. TensorDistribution makes this natural.

**Selecting distributions based on conditions:**

```python
# You have distributions for different environments
env_dists = TensorNormal(
    loc=torch.randn(100, 4),            # 100 environments, 4D actions  
    scale=torch.ones(100, 4),
    shape=(100,),
    device='cpu'
)

# Select distributions for high-performing environments
high_performance_mask = torch.rand(100) > 0.8
good_env_dists = env_dists[high_performance_mask]

# Sample only from the good environments
good_actions = good_env_dists.sample()  # Shape: (num_good_envs, 4)

# Get distributions for specific environments
specific_envs = env_dists[torch.tensor([5, 12, 23, 47])]
specific_actions = specific_envs.sample()  # Shape: (4, 4)
```

This indexing works exactly like tensor indexing, making it intuitive for anyone familiar with PyTorch.

### 4. Seamless Stacking and Concatenation

Combining distributions from different sources is common in ML workflows. TensorDistribution supports `torch.stack` and `torch.cat` operations naturally.

**Example:** Combining distributions from different policy networks:

```python
# Two different policies produce distributions
policy_a_dist = TensorNormal(
    loc=torch.randn(32, 4),
    scale=torch.ones(32, 4),
    shape=(32,),
    device='cpu'
)

policy_b_dist = TensorNormal(
    loc=torch.randn(32, 4),
    scale=torch.ones(32, 4) * 0.5,      # Different scale
    shape=(32,),
    device='cpu'
)

# Stack to compare policies side by side
compared_policies = torch.stack([policy_a_dist, policy_b_dist], dim=0)
assert compared_policies.shape == (2, 32)

# Sample from both policies
policy_samples = compared_policies.sample()  # Shape: (2, 32, 4)

# Concatenate to create a larger batch
combined_batch = torch.cat([policy_a_dist, policy_b_dist], dim=0)
assert combined_batch.shape == (64,)
```

## Common Workflows

### Pattern 1: Training with Batched Distributions

One of the most common patterns is training models that output probability distributions. Here's how TensorDistribution simplifies the process:

**Scenario:** You're training a variational autoencoder where the encoder outputs parameters for latent distributions.

```python
# Your encoder outputs these parameters for a batch of inputs
batch_size = 64
latent_dim = 20

mu = torch.randn(batch_size, latent_dim, requires_grad=True)
log_var = torch.randn(batch_size, latent_dim, requires_grad=True)
std = torch.exp(0.5 * log_var)

# Create latent distribution
latent_dist = TensorNormal(
    loc=mu,
    scale=std,
    shape=(batch_size,),
    device='cpu'
)

# Sample latent codes (reparameterization trick for gradients)
latent_codes = latent_dist.rsample()    # Shape: (batch_size, latent_dim)

# Compute KL divergence with prior
prior = TensorNormal(
    loc=torch.zeros(batch_size, latent_dim),
    scale=torch.ones(batch_size, latent_dim),
    shape=(batch_size,),
    device='cpu'
)

kl_divergence = torch.distributions.kl_divergence(latent_dist, prior)
kl_loss = kl_divergence.sum(dim=1).mean()  # Average over batch

# The gradients flow back through the distribution parameters automatically
kl_loss.backward()
```

The key advantage here is that gradients flow naturally through the TensorDistribution, and you can compute KL divergence with a single operation across the entire batch.

### Pattern 2: Hierarchical Distributions

Many models involve hierarchical structures where you have distributions at different levels. TensorDistribution makes this natural to work with.

**Scenario:** You have a hierarchical model where each group has its own distribution, and within each group, individual items have their own distributions.

```python
# Group-level parameters (5 groups)
group_means = torch.randn(5, 1, 8)     # Shared within each group
group_scales = torch.ones(5, 1, 8)

# Individual-level noise (10 items per group)
individual_noise = torch.randn(5, 10, 8) * 0.1

# Create hierarchical distribution
hierarchical_dist = TensorNormal(
    loc=group_means + individual_noise,
    scale=group_scales,
    shape=(5, 10),                      # 5 groups, 10 items each
    device='cpu'
)

# Sample from the hierarchy
samples = hierarchical_dist.sample()   # Shape: (5, 10, 8)

# You can easily work with individual groups
group_0_dist = hierarchical_dist[0]    # Shape: (10,) - just group 0
group_0_samples = group_0_dist.sample() # Shape: (10, 8)

# Or reshape to work with all items as a flat batch
flat_dist = hierarchical_dist.view(-1)  # Shape: (50,)
all_samples = flat_dist.sample()        # Shape: (50, 8)
```

### Pattern 3: Dynamic Distribution Selection

Sometimes you need to work with different types of distributions based on runtime conditions. TensorDistribution makes this manageable.

**Scenario:** Your model switches between continuous and discrete action spaces based on the environment.

```python
def create_action_distribution(action_type, batch_size, action_dim):
    """Create appropriate distribution based on action type."""
    if action_type == 'continuous':
        # Continuous actions - use Normal distribution
        return TensorNormal(
            loc=torch.zeros(batch_size, action_dim),
            scale=torch.ones(batch_size, action_dim),
            shape=(batch_size,),
            device='cpu'
        )
    else:
        # Discrete actions - use Categorical distribution  
        from tensorcontainer.tensor_distribution import TensorCategorical
        return TensorCategorical(
            logits=torch.randn(batch_size, action_dim),
            shape=(batch_size,),
            device='cpu'
        )

# Usage in your training loop
for batch_data, env_types in dataloader:
    for env_type in ['continuous', 'discrete']:
        # Create appropriate distribution
        action_dist = create_action_distribution(env_type, 32, 6)
        
        # The same sampling interface works for both types
        actions = action_dist.sample()
        log_probs = action_dist.log_prob(actions)
        
        # Tensor operations work the same way
        gpu_dist = action_dist.to('cuda')
        reshaped_dist = action_dist.view(8, 4)
```

The power here is that once you have a TensorDistribution, the interface is consistent regardless of the underlying distribution type.

## Integration with TensorContainer

The real power of TensorDistribution emerges when you combine it with other TensorContainer types like TensorDict and TensorDataClass. This creates a unified system where entire data structures can be manipulated with simple tensor operations.

### Working with TensorDict

When you have multiple related distributions, TensorDict provides a clean way to organize them:

```python
from tensorcontainer import TensorDict
from tensorcontainer.tensor_distribution import TensorNormal, TensorCategorical

# Create a structured collection of distributions
agent_distributions = TensorDict({
    'policy': TensorNormal(
        loc=torch.zeros(32, 6),     # 6-dimensional continuous actions
        scale=torch.ones(32, 6),
        shape=(32,),
        device='cpu'
    ),
    'value': TensorNormal(
        loc=torch.zeros(32, 1),     # 1-dimensional value estimates
        scale=torch.ones(32, 1),
        shape=(32,),
        device='cpu'
    ),
    'attention': TensorCategorical(
        logits=torch.randn(32, 10), # 10 possible attention targets
        shape=(32,),
        device='cpu'
    )
}, shape=(32,), device='cpu')

# Move all distributions to GPU with one operation
gpu_distributions = agent_distributions.to('cuda')

# Reshape all distributions together
reshaped_distributions = agent_distributions.view(8, 4)

# Sample from all distributions at once
samples = {
    key: dist.sample() 
    for key, dist in agent_distributions.items()
}
```

This is incredibly powerful for complex models where you have many related distributions that need to be managed together.

### Type-Safe Distribution Structures with TensorDataClass

For production code where type safety matters, you can use TensorDataClass to create structured distribution objects:

```python
from tensorcontainer import TensorDataClass

class PolicyOutputs(TensorDataClass):
    action_dist: TensorNormal
    value_dist: TensorNormal
    done_prob: TensorBernoulli
    
# Create structured policy outputs
policy_outputs = PolicyOutputs(
    action_dist=TensorNormal(
        loc=torch.randn(32, 4),
        scale=torch.ones(32, 4),
        shape=(32,),
        device='cpu'
    ),
    value_dist=TensorNormal(
        loc=torch.randn(32, 1),
        scale=torch.ones(32, 1),
        shape=(32,),
        device='cpu'
    ),
    done_prob=TensorBernoulli(
        probs=torch.sigmoid(torch.randn(32, 1)),
        shape=(32,),
        device='cpu'
    ),
    shape=(32,),
    device='cpu'
)

# All tensor operations work on the entire structure
gpu_outputs = policy_outputs.to('cuda')
reshaped_outputs = policy_outputs.view(8, 4)

# Access individual distributions with full IDE support
actions = policy_outputs.action_dist.sample()
values = policy_outputs.value_dist.sample()
done_flags = policy_outputs.done_prob.sample()
```

This gives you the best of both worlds: the convenience of tensor-like operations and the safety of static typing.

## Why This Matters

TensorDistribution isn't just about convenience - it solves real problems that make probabilistic modeling more reliable and maintainable:

1. **Fewer Bugs:** When you manually manage distribution parameters, it's easy to forget to update one when moving devices or reshaping. TensorDistribution eliminates these bugs by keeping everything synchronized.

2. **Cleaner Code:** Your code focuses on the logic of your model rather than the mechanics of parameter management. Compare 50 lines of manual parameter handling with 2 lines of TensorDistribution operations.

3. **Better Performance:** By leveraging PyTorch's tensor operations, TensorDistribution can apply transformations more efficiently than manual loops over individual distributions.

4. **Easier Debugging:** When all your distributions behave like tensors, you can use the same debugging tools and mental models you already know from working with tensors.

5. **Future-Proof:** As your models grow more complex, TensorDistribution scales with you. Adding new distribution types or nested structures doesn't require rewriting your core tensor operations.

TensorDistribution transforms probability distributions from special objects requiring careful handling into first-class citizens of the PyTorch tensor ecosystem. This makes probabilistic modeling feel as natural as regular neural network development.