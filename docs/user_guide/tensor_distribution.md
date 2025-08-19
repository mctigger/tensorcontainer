# TensorDistribution Deep Dive

TensorDistribution bridges the gap between PyTorch's `torch.distributions` and TensorContainer's tensor-like operations. This guide covers distribution types, sampling strategies, and integration patterns that make TensorDistribution essential for probabilistic modeling and reinforcement learning.

## Distribution Types and Creation

### Continuous Distributions

```python
import torch
from tensorcontainer.tensor_distribution import (
    TensorNormal, TensorBeta, TensorGamma, TensorExponential,
    TensorTruncatedNormal, TensorTanhNormal
)

# Normal (Gaussian) distribution
normal = TensorNormal(
    loc=torch.zeros(32, 4),       # Mean
    scale=torch.ones(32, 4),      # Standard deviation
    shape=(32,),
    device='cpu'
)

# Beta distribution (values in [0, 1])
beta = TensorBeta(
    concentration1=torch.ones(32, 2),  # Alpha parameter
    concentration0=torch.ones(32, 2),  # Beta parameter  
    shape=(32,),
    device='cpu'
)

# Truncated normal (bounded continuous)
truncated_normal = TensorTruncatedNormal(
    loc=torch.zeros(32, 2),
    scale=torch.ones(32, 2),
    low=-1.0,   # Lower bound
    high=1.0,   # Upper bound
    shape=(32,),
    device='cpu'
)

# Tanh-transformed normal (squashed to [-1, 1])
tanh_normal = TensorTanhNormal(
    loc=torch.zeros(32, 2),
    scale=torch.ones(32, 2),
    shape=(32,),
    device='cpu'
)
```

### Discrete Distributions

```python
from tensorcontainer.tensor_distribution import (
    TensorCategorical, TensorBernoulli, TensorBinomial,
    TensorOneHotCategorical, TensorPoisson
)

# Categorical distribution (discrete choice)
categorical = TensorCategorical(
    logits=torch.randn(32, 6),    # Raw scores for 6 categories
    shape=(32,),
    device='cpu'
)

# Alternative: from probabilities
categorical_probs = TensorCategorical(
    probs=torch.softmax(torch.randn(32, 6), dim=-1),
    shape=(32,),
    device='cpu'
)

# Bernoulli distribution (binary)
bernoulli = TensorBernoulli(
    probs=torch.sigmoid(torch.randn(32, 1)),
    shape=(32,),
    device='cpu'
)

# One-hot categorical (returns one-hot vectors)
one_hot = TensorOneHotCategorical(
    logits=torch.randn(32, 6),
    shape=(32,),
    device='cpu'
)

# Binomial distribution
binomial = TensorBinomial(
    total_count=10,                    # Number of trials
    probs=torch.rand(32, 1) * 0.5,    # Success probability
    shape=(32,),
    device='cpu'
)
```

### Multivariate Distributions

```python
from tensorcontainer.tensor_distribution import (
    TensorMultivariateNormal, TensorDirichlet, 
    TensorLowRankMultivariateNormal
)

# Multivariate normal
mv_normal = TensorMultivariateNormal(
    loc=torch.zeros(32, 3),                    # Mean vector
    covariance_matrix=torch.eye(3).expand(32, 3, 3),  # Covariance
    shape=(32,),
    device='cpu'
)

# Low-rank multivariate normal (efficient for high dimensions)
low_rank_mv_normal = TensorLowRankMultivariateNormal(
    loc=torch.zeros(32, 10),
    cov_factor=torch.randn(32, 10, 3),        # Low-rank factor
    cov_diag=torch.ones(32, 10),              # Diagonal component
    shape=(32,),
    device='cpu'
)

# Dirichlet distribution (probability simplex)
dirichlet = TensorDirichlet(
    concentration=torch.ones(32, 5),          # Concentration parameters
    shape=(32,),
    device='cpu'
)
```

## Sampling and Probability Computation

### Basic Operations

```python
# Sampling
samples = normal.sample()                    # Shape: (32, 4)
multiple_samples = normal.sample((10,))      # Shape: (10, 32, 4)

# Probability density/mass
log_probs = normal.log_prob(samples)         # Shape: (32, 4)
probs = torch.exp(log_probs)                 # Actual probabilities

# Statistical properties
mean = normal.mean                           # Shape: (32, 4)
variance = normal.variance                   # Shape: (32, 4)
entropy = normal.entropy()                   # Shape: (32, 4)
```

### Advanced Sampling Strategies

```python
# Reparameterized sampling (for gradients)
normal_rsample = normal.rsample()            # Differentiable sample
normal_rsample.backward()                    # Gradients flow through

# Sample with custom shapes
batch_samples = normal.sample((5, 3))        # Shape: (5, 3, 32, 4)

# Constrained sampling for discrete distributions
def sample_with_temperature(dist, temperature=1.0):
    """Sample from distribution with temperature scaling."""
    if hasattr(dist, '_logits'):
        # Scale logits by temperature
        scaled_logits = dist._logits / temperature
        temp_dist = TensorCategorical(logits=scaled_logits, shape=dist.shape)
        return temp_dist.sample()
    else:
        return dist.sample()

# Usage
hot_samples = sample_with_temperature(categorical, temperature=0.5)   # More peaked
cold_samples = sample_with_temperature(categorical, temperature=2.0)  # More uniform
```

### Probability Computation Patterns

```python
# Log probability computation for different action types
def compute_log_probs(dist, actions):
    """Compute log probabilities handling different action types."""
    if isinstance(dist, TensorCategorical):
        # Discrete actions: use gather or direct indexing
        return dist.log_prob(actions.long())
    elif isinstance(dist, TensorNormal):
        # Continuous actions: direct log_prob
        return dist.log_prob(actions).sum(dim=-1)  # Sum over action dims
    elif isinstance(dist, TensorOneHotCategorical):
        # One-hot actions: convert to indices first
        action_indices = actions.argmax(dim=-1)
        return dist.log_prob(actions)
    else:
        return dist.log_prob(actions)

# Likelihood ratios for importance sampling
def importance_weights(new_dist, old_dist, actions):
    """Compute importance sampling weights."""
    new_log_probs = compute_log_probs(new_dist, actions)
    old_log_probs = compute_log_probs(old_dist, actions)
    return torch.exp(new_log_probs - old_log_probs)
```

## Tensor-Like Operations

### Shape and Device Operations

```python
# All standard tensor operations work
gpu_dist = normal.to('cuda')
reshaped_dist = normal.reshape(8, 4)
squeezed_dist = normal.squeeze()
expanded_dist = normal.expand(64, -1)

# Indexing and slicing
first_sample_dist = normal[0]                # Distribution for first sample
subset_dist = normal[:16]                    # First 16 samples
masked_dist = normal[mask]                   # Boolean masking

# Concatenation and stacking
stacked_dists = torch.stack([normal, normal])  # New batch dimension
catted_dists = torch.cat([normal, normal])     # Extend batch dimension
```

### Advanced Transformations

```python
# Gradients and optimization
normal_with_grads = TensorNormal(
    loc=torch.zeros(32, 4, requires_grad=True),
    scale=torch.ones(32, 4, requires_grad=True),
    shape=(32,),
    device='cpu'
)

# Loss computation with gradients
samples = normal_with_grads.rsample()
loss = (samples ** 2).mean()
loss.backward()

print(f"Loc gradients: {normal_with_grads._loc.grad}")
print(f"Scale gradients: {normal_with_grads._scale.grad}")

# Detachment for stop-gradient operations
detached_dist = normal_with_grads.detach()
detached_samples = detached_dist.sample()    # No gradients
```

### Distribution Arithmetic

```python
# KL divergence between distributions
from torch.distributions import kl_divergence

kl_div = kl_divergence(normal, detached_dist)
print(f"KL divergence shape: {kl_div.shape}")  # (32, 4)

# Custom divergence computations
def js_divergence(p, q):
    """Jensen-Shannon divergence between two distributions."""
    m_samples = 0.5 * (p.sample() + q.sample())
    
    kl_pm = kl_divergence(p, m_samples)
    kl_qm = kl_divergence(q, m_samples)
    
    return 0.5 * (kl_pm + kl_qm)

# Mixture distributions
def create_mixture(dists, weights):
    """Create mixture of distributions."""
    # Sample component indices
    component_dist = TensorCategorical(logits=torch.log(weights))
    components = component_dist.sample()
    
    # Sample from selected components
    samples = torch.stack([d.sample() for d in dists])
    selected_samples = samples[components, torch.arange(len(components))]
    
    return selected_samples
```

## Reinforcement Learning Integration

### Policy Networks

```python
import torch.nn as nn

class ContinuousPolicyNetwork(nn.Module):
    """Policy network outputting continuous action distributions."""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, observations):
        features = self.backbone(observations)
        
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        std = torch.exp(log_std.clamp(-20, 2))  # Numerical stability
        
        return TensorNormal(
            loc=mean,
            scale=std,
            shape=observations.shape[:1],  # Batch shape
            device=observations.device
        )

class DiscretePolicyNetwork(nn.Module):
    """Policy network outputting discrete action distributions."""
    
    def __init__(self, obs_dim, num_actions, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, observations):
        logits = self.network(observations)
        
        return TensorCategorical(
            logits=logits,
            shape=observations.shape[:1],
            device=observations.device
        )

# Usage
obs = torch.randn(32, 128)

continuous_policy = ContinuousPolicyNetwork(128, 4)
discrete_policy = DiscretePolicyNetwork(128, 6)

action_dist_continuous = continuous_policy(obs)
action_dist_discrete = discrete_policy(obs)

# Sample actions
continuous_actions = action_dist_continuous.sample()  # Shape: (32, 4)
discrete_actions = action_dist_discrete.sample()      # Shape: (32,)
```

### Value Function Training

```python
class DistributionalValueNetwork(nn.Module):
    """Value network using distributional RL."""
    
    def __init__(self, obs_dim, num_atoms=51, v_min=-10, v_max=10):
        super().__init__()
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_atoms)
        )
        
        # Support atoms for value distribution
        self.register_buffer('support', torch.linspace(v_min, v_max, num_atoms))
    
    def forward(self, observations):
        logits = self.network(observations)
        
        return TensorCategorical(
            logits=logits,
            shape=observations.shape[:1],
            device=observations.device
        )
    
    def get_values(self, observations):
        """Get expected values from distribution."""
        value_dist = self.forward(observations)
        probs = torch.softmax(value_dist._logits, dim=-1)
        return (probs * self.support).sum(dim=-1, keepdim=True)

# Training with distributional loss
def distributional_loss(pred_dist, target_values, gamma=0.99):
    """Compute distributional RL loss."""
    # Project target values onto support
    batch_size = target_values.shape[0]
    support = pred_dist.support if hasattr(pred_dist, 'support') else torch.linspace(-10, 10, 51)
    
    # Bellman target distribution
    target_support = target_values + gamma * support.unsqueeze(0)
    target_probs = torch.zeros(batch_size, len(support))
    
    # Project onto support (simplified)
    for i in range(batch_size):
        target_probs[i] = torch.softmax(target_support[i], dim=0)
    
    # Cross-entropy loss
    pred_log_probs = pred_dist.log_prob(torch.arange(len(support)).float())
    return -torch.sum(target_probs * pred_log_probs, dim=-1).mean()
```

### PPO and Policy Optimization

```python
def ppo_loss(old_dist, new_dist, actions, advantages, clip_epsilon=0.2):
    """Proximal Policy Optimization loss."""
    # Compute probability ratios
    old_log_probs = old_dist.log_prob(actions)
    new_log_probs = new_dist.log_prob(actions)
    
    # Handle multi-dimensional actions
    if len(old_log_probs.shape) > 1:
        old_log_probs = old_log_probs.sum(dim=-1)
        new_log_probs = new_log_probs.sum(dim=-1)
    
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # Clipped objective
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    
    policy_loss = -torch.min(
        ratio * advantages,
        clipped_ratio * advantages
    ).mean()
    
    return policy_loss

def entropy_bonus(dist, coefficient=0.01):
    """Entropy regularization for exploration."""
    entropy = dist.entropy()
    if len(entropy.shape) > 1:
        entropy = entropy.sum(dim=-1)
    return coefficient * entropy.mean()

# Training loop integration
def train_policy(policy, old_policy, observations, actions, advantages):
    """Train policy with PPO."""
    # Get new action distribution
    new_action_dist = policy(observations)
    
    # Get old action distribution (detached)
    with torch.no_grad():
        old_action_dist = old_policy(observations).detach()
    
    # Compute losses
    policy_loss = ppo_loss(old_action_dist, new_action_dist, actions, advantages)
    entropy_loss = entropy_bonus(new_action_dist)
    
    total_loss = policy_loss - entropy_loss
    
    return total_loss, {
        'policy_loss': policy_loss.item(),
        'entropy': entropy_loss.item()
    }
```

## Advanced Applications

### Bayesian Neural Networks

```python
class BayesianLinear(nn.Module):
    """Bayesian linear layer with weight distributions."""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        
        # Weight distribution parameters
        self.weight_mean = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_log_std = nn.Parameter(torch.full((out_features, in_features), -3.0))
        
        # Bias distribution parameters  
        self.bias_mean = nn.Parameter(torch.zeros(out_features))
        self.bias_log_std = nn.Parameter(torch.full((out_features,), -3.0))
    
    def forward(self, x, sample=True):
        if sample:
            # Sample weights and biases
            weight_dist = TensorNormal(
                loc=self.weight_mean,
                scale=torch.exp(self.weight_log_std),
                shape=self.weight_mean.shape[:1],
                device=x.device
            )
            
            bias_dist = TensorNormal(
                loc=self.bias_mean,
                scale=torch.exp(self.bias_log_std),
                shape=self.bias_mean.shape[:1],
                device=x.device
            )
            
            weight = weight_dist.rsample()
            bias = bias_dist.rsample()
        else:
            # Use mean values
            weight = self.weight_mean
            bias = self.bias_mean
        
        return F.linear(x, weight, bias)

# Uncertainty quantification
def predict_with_uncertainty(model, x, num_samples=100):
    """Get predictions with uncertainty estimates."""
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for _ in range(num_samples):
            pred = model(x, sample=True)
            predictions.append(pred)
    
    predictions = torch.stack(predictions)
    
    mean_prediction = predictions.mean(dim=0)
    uncertainty = predictions.std(dim=0)
    
    return mean_prediction, uncertainty
```

### Generative Models

```python
class VAE(nn.Module):
    """Variational Autoencoder with TensorDistribution."""
    
    def __init__(self, input_dim, latent_dim, hidden_dim=256):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.encoder_mean = nn.Linear(hidden_dim, latent_dim)
        self.encoder_log_std = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode input to latent distribution."""
        features = self.encoder(x)
        
        mean = self.encoder_mean(features)
        log_std = self.encoder_log_std(features)
        std = torch.exp(log_std.clamp(-20, 2))
        
        return TensorNormal(
            loc=mean,
            scale=std,
            shape=x.shape[:1],
            device=x.device
        )
    
    def decode(self, z):
        """Decode latent variables to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x):
        # Encode to latent distribution
        latent_dist = self.encode(x)
        
        # Sample latent variables
        z = latent_dist.rsample()
        
        # Decode to reconstruction
        reconstruction = self.decode(z)
        
        return reconstruction, latent_dist

def vae_loss(recon, target, latent_dist, beta=1.0):
    """VAE loss with KL divergence regularization."""
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(recon, target, reduction='mean')
    
    # KL divergence with standard normal prior
    prior = TensorNormal(
        loc=torch.zeros_like(latent_dist._loc),
        scale=torch.ones_like(latent_dist._scale),
        shape=latent_dist.shape,
        device=latent_dist.device
    )
    
    kl_loss = kl_divergence(latent_dist, prior).sum(dim=-1).mean()
    
    return recon_loss + beta * kl_loss
```

## Best Practices

### Performance Optimization

1. **Batch Operations**: Apply operations to entire distributions rather than individual samples
2. **Memory Efficiency**: Use `rsample()` for gradients, `sample()` for inference
3. **Numerical Stability**: Clamp log_std parameters and use stable implementations
4. **Device Consistency**: Keep distribution parameters on the same device as input data

### Design Patterns

1. **Parameter Sharing**: Use shared networks for distribution parameters when appropriate
2. **Temperature Scaling**: Implement temperature parameters for controlling randomness
3. **Gradient Flow**: Be aware of when gradients flow through samples vs. parameters
4. **Validation**: Always validate distribution parameters during development

### Integration Guidelines

1. **Model Interfaces**: Design models to return TensorDistribution instances directly
2. **Loss Functions**: Implement custom loss functions that work with distribution objects
3. **Metric Tracking**: Monitor entropy, KL divergence, and other distributional properties
4. **Debugging**: Use distribution properties (mean, variance) for debugging and validation

TensorDistribution excels in probabilistic modeling scenarios where you need the flexibility of PyTorch distributions combined with the operational convenience of tensor-like objects. Its seamless integration with TensorContainer operations makes it particularly valuable for reinforcement learning, Bayesian modeling, and other uncertainty-aware applications.
