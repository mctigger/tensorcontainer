# TensorDataClass Deep Dive

TensorDataClass provides a dataclass-based approach to tensor containers with static typing, automatic field generation, and comprehensive IDE support. This guide covers advanced field definitions, inheritance patterns, and integration strategies that make TensorDataClass ideal for production workflows.

## Class Definition and Field Types

### Basic Field Definitions

```python
import torch
from tensorcontainer import TensorDataClass

class MLBatch(TensorDataClass):
    # Required tensor fields
    features: torch.Tensor
    labels: torch.Tensor
    
    # TensorDataClass automatically:
    # - Creates __init__ with these fields + shape/device
    # - Enables field access via dot notation
    # - Provides dataclass features (repr, etc.)
    # - Validates tensor shapes against batch dimensions

# Usage
batch = MLBatch(
    features=torch.randn(32, 128),
    labels=torch.randint(0, 10, (32,)),
    shape=(32,),  # Required: batch dimensions
    device='cpu'  # Optional: device constraint
)

# Type-safe field access
features = batch.features  # IDE knows this is torch.Tensor
batch.labels = torch.randint(0, 5, (32,))  # Type-checked assignment
```

### Advanced Field Types with dataclasses.field

```python
from dataclasses import field
from typing import Optional, Dict, List, Any, Union

class AdvancedBatch(TensorDataClass):
    # Required tensor fields
    observations: torch.Tensor
    actions: torch.Tensor
    
    # Optional tensor fields
    next_observations: Optional[torch.Tensor] = None
    rewards: Optional[torch.Tensor] = None
    
    # Non-tensor metadata fields
    episode_ids: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    algorithm_config: Dict[str, Union[int, float, str]] = field(default_factory=dict)
    
    # Tensor fields with default values
    done: torch.Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.bool))
    info: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    
    # Private fields (not processed by TensorContainer operations)
    _internal_state: Any = field(default=None, init=False)

# Usage with partial initialization
batch = AdvancedBatch(
    observations=torch.randn(16, 84, 84, 3),
    actions=torch.randint(0, 4, (16,)),
    episode_ids=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
    shape=(16,),
    device='cpu'
)

# Add optional fields later
batch.rewards = torch.randn(16, 1)
batch.next_observations = torch.randn(16, 84, 84, 3)
```

### Field Validation and Constraints

```python
from dataclasses import field
import torch.nn.functional as F

class ValidatedBatch(TensorDataClass):
    probabilities: torch.Tensor  # Should sum to 1 along last dim
    logits: torch.Tensor        # Raw network outputs
    
    def __post_init__(self):
        super().__post_init__()  # Always call parent first
        
        # Custom validation logic
        if self.probabilities is not None:
            prob_sums = self.probabilities.sum(dim=-1)
            if not torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5):
                raise ValueError("Probabilities must sum to 1 along last dimension")
        
        # Ensure logits and probabilities are consistent
        if self.logits is not None and self.probabilities is not None:
            expected_probs = F.softmax(self.logits, dim=-1)
            if not torch.allclose(self.probabilities, expected_probs, atol=1e-5):
                raise ValueError("Probabilities don't match softmax of logits")

# Usage with validation
logits = torch.randn(32, 10)
probs = F.softmax(logits, dim=-1)

validated_batch = ValidatedBatch(
    probabilities=probs,
    logits=logits,
    shape=(32,),
    device='cpu'
)
```

## Inheritance and Composition

### Single Inheritance

```python
class BaseBatch(TensorDataClass):
    """Base class for all RL batches."""
    observations: torch.Tensor
    actions: torch.Tensor

class PolicyBatch(BaseBatch):
    """Extends base with policy-specific fields."""
    log_probs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor

class OffPolicyBatch(BaseBatch):
    """Extends base with off-policy specific fields."""
    next_observations: torch.Tensor
    rewards: torch.Tensor
    done: torch.Tensor
    q_values: torch.Tensor

# All parent fields are available
policy_batch = PolicyBatch(
    observations=torch.randn(32, 128),
    actions=torch.randn(32, 4),
    log_probs=torch.randn(32, 4),
    values=torch.randn(32, 1),
    advantages=torch.randn(32, 1),
    shape=(32,),
    device='cpu'
)
```

### Multiple Inheritance Patterns

```python
class TimestampedMixin(TensorDataClass):
    """Mixin for adding timestamp information."""
    timestamps: torch.Tensor
    
class RewardMixin(TensorDataClass):
    """Mixin for adding reward information."""
    rewards: torch.Tensor
    discounted_rewards: torch.Tensor

class CompleteRLBatch(BaseBatch, TimestampedMixin, RewardMixin):
    """Combines multiple mixins with base functionality."""
    values: torch.Tensor

# All fields from all parent classes are available
complete_batch = CompleteRLBatch(
    # From BaseBatch
    observations=torch.randn(32, 128),
    actions=torch.randn(32, 4),
    # From TimestampedMixin
    timestamps=torch.arange(32, dtype=torch.float),
    # From RewardMixin  
    rewards=torch.randn(32, 1),
    discounted_rewards=torch.randn(32, 1),
    # From CompleteRLBatch
    values=torch.randn(32, 1),
    shape=(32,),
    device='cpu'
)
```

### Abstract Base Classes

```python
from abc import ABC, abstractmethod

class AbstractBatch(TensorDataClass, ABC):
    """Abstract base for defining batch interfaces."""
    observations: torch.Tensor
    
    @abstractmethod
    def compute_loss(self) -> torch.Tensor:
        """Compute loss for this batch type."""
        pass
    
    @abstractmethod
    def get_action_dim(self) -> int:
        """Return the action dimensionality."""
        pass

class DiscreteBatch(AbstractBatch):
    """Concrete implementation for discrete action spaces."""
    actions: torch.Tensor  # Shape: (batch_size,)
    
    def compute_loss(self) -> torch.Tensor:
        # Implement discrete action loss
        return F.cross_entropy(self.logits, self.actions)
    
    def get_action_dim(self) -> int:
        return self.actions.max().item() + 1

class ContinuousBatch(AbstractBatch):
    """Concrete implementation for continuous action spaces."""
    actions: torch.Tensor  # Shape: (batch_size, action_dim)
    
    def compute_loss(self) -> torch.Tensor:
        # Implement continuous action loss
        return F.mse_loss(self.predicted_actions, self.actions)
    
    def get_action_dim(self) -> int:
        return self.actions.shape[-1]
```

## Advanced Usage Patterns

### Factory Methods and Class Methods

```python
class ExperienceBatch(TensorDataClass):
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    done: torch.Tensor
    
    @classmethod
    def from_trajectory(cls, trajectory: List[Dict[str, torch.Tensor]], 
                       device: str = 'cpu'):
        """Create batch from a list of trajectory steps."""
        states = torch.stack([step['state'] for step in trajectory])
        actions = torch.stack([step['action'] for step in trajectory])
        rewards = torch.stack([step['reward'] for step in trajectory])
        next_states = torch.stack([step['next_state'] for step in trajectory])
        done = torch.stack([step['done'] for step in trajectory])
        
        return cls(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            done=done,
            shape=(len(trajectory),),
            device=device
        )
    
    @classmethod
    def empty(cls, batch_size: int, state_dim: int, action_dim: int, 
              device: str = 'cpu'):
        """Create empty batch with specified dimensions."""
        return cls(
            states=torch.empty(batch_size, state_dim, device=device),
            actions=torch.empty(batch_size, action_dim, device=device),
            rewards=torch.empty(batch_size, 1, device=device),
            next_states=torch.empty(batch_size, state_dim, device=device),
            done=torch.empty(batch_size, 1, dtype=torch.bool, device=device),
            shape=(batch_size,),
            device=device
        )
    
    def split_episodes(self) -> List['ExperienceBatch']:
        """Split batch by episode boundaries based on done flags."""
        episode_starts = torch.cat([
            torch.tensor([0], device=self.device),
            (self.done[:-1] == True).nonzero(as_tuple=True)[0] + 1
        ])
        episode_ends = torch.cat([
            (self.done == True).nonzero(as_tuple=True)[0] + 1,
            torch.tensor([len(self.done)], device=self.device)
        ])
        
        episodes = []
        for start, end in zip(episode_starts, episode_ends):
            episodes.append(self[start:end])
        
        return episodes

# Usage
trajectory_data = [
    {'state': torch.randn(128), 'action': torch.randn(4), 
     'reward': torch.tensor(1.0), 'next_state': torch.randn(128), 
     'done': torch.tensor(False)},
    # ... more steps
]

batch = ExperienceBatch.from_trajectory(trajectory_data, device='cuda')
empty_batch = ExperienceBatch.empty(32, 128, 4, device='cuda')
episodes = batch.split_episodes()
```

### Integration with Neural Networks

```python
import torch.nn as nn

class NetworkBatch(TensorDataClass):
    """Batch optimized for neural network processing."""
    inputs: torch.Tensor
    targets: torch.Tensor
    weights: Optional[torch.Tensor] = None
    
    def apply_network(self, network: nn.Module) -> torch.Tensor:
        """Apply network to batch inputs."""
        return network(self.inputs)
    
    def compute_weighted_loss(self, predictions: torch.Tensor, 
                            loss_fn: nn.Module) -> torch.Tensor:
        """Compute loss with optional sample weights."""
        loss = loss_fn(predictions, self.targets)
        
        if self.weights is not None:
            loss = loss * self.weights.squeeze()
        
        return loss.mean()

class PolicyNetworkBatch(NetworkBatch):
    """Specialized for policy network training."""
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    
    def compute_ppo_loss(self, new_log_probs: torch.Tensor, 
                        clip_epsilon: float = 0.2) -> torch.Tensor:
        """Compute PPO clipped objective loss."""
        ratio = torch.exp(new_log_probs - self.old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        
        loss = -torch.min(
            ratio * self.advantages,
            clipped_ratio * self.advantages
        ).mean()
        
        return loss

# Usage with models
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

model = PolicyNetwork(128, 4)
batch = PolicyNetworkBatch(
    inputs=torch.randn(32, 128),
    targets=torch.randn(32, 4),
    actions=torch.randn(32, 4),
    old_log_probs=torch.randn(32),
    advantages=torch.randn(32),
    shape=(32,),
    device='cpu'
)

predictions = batch.apply_network(model)
loss = batch.compute_ppo_loss(predictions)
```

### Serialization and State Management

```python
import pickle
from pathlib import Path

class SerializableBatch(TensorDataClass):
    data: torch.Tensor
    labels: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save batch to disk."""
        path = Path(path)
        
        # Save tensors separately for efficiency
        torch.save({
            'data': self.data,
            'labels': self.labels,
            'shape': self.shape,
            'device': str(self.device)
        }, path.with_suffix('.pt'))
        
        # Save metadata separately  
        with open(path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(self.metadata, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'SerializableBatch':
        """Load batch from disk."""
        path = Path(path)
        
        # Load tensors
        tensor_data = torch.load(path.with_suffix('.pt'))
        
        # Load metadata
        with open(path.with_suffix('.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        return cls(
            data=tensor_data['data'],
            labels=tensor_data['labels'],
            metadata=metadata,
            shape=tensor_data['shape'],
            device=tensor_data['device']
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'data': self.data.cpu().numpy().tolist(),
            'labels': self.labels.cpu().numpy().tolist(),
            'metadata': self.metadata,
            'shape': list(self.shape),
            'device': str(self.device)
        }
    
    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> 'SerializableBatch':
        """Create from dictionary."""
        return cls(
            data=torch.tensor(data_dict['data']),
            labels=torch.tensor(data_dict['labels']),
            metadata=data_dict['metadata'],
            shape=tuple(data_dict['shape']),
            device=data_dict['device']
        )

# Usage
batch = SerializableBatch(
    data=torch.randn(16, 64),
    labels=torch.randint(0, 10, (16,)),
    metadata={'experiment': 'run_1', 'epoch': 5},
    shape=(16,),
    device='cpu'
)

# Save and load
batch.save('experiment_data')
loaded_batch = SerializableBatch.load('experiment_data')

# JSON serialization
import json
data_dict = batch.to_dict()
json_str = json.dumps(data_dict)
restored_batch = SerializableBatch.from_dict(json.loads(json_str))
```

## Memory Optimization and Performance

### Slots and Memory Efficiency

```python
# TensorDataClass automatically uses slots=True for memory efficiency
class EfficientBatch(TensorDataClass):
    features: torch.Tensor
    labels: torch.Tensor
    
    # With slots=True (automatic), instances use less memory
    # and have faster attribute access

# Memory comparison
import sys

# Regular class
class RegularClass:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

# TensorDataClass with slots
batch_data = torch.randn(1000, 128)
batch_labels = torch.randint(0, 10, (1000,))

regular = RegularClass(batch_data, batch_labels)
efficient = EfficientBatch(batch_data, batch_labels, shape=(1000,), device='cpu')

print(f"Regular class size: {sys.getsizeof(regular)} bytes")
print(f"TensorDataClass size: {sys.getsizeof(efficient)} bytes")
# TensorDataClass typically uses 30-50% less memory
```

### Lazy Loading and Computed Properties

```python
class LazyBatch(TensorDataClass):
    raw_data: torch.Tensor
    _processed_cache: Optional[torch.Tensor] = field(default=None, init=False)
    
    @property
    def processed_data(self) -> torch.Tensor:
        """Lazily compute and cache processed data."""
        if self._processed_cache is None:
            # Expensive computation only done once
            self._processed_cache = F.normalize(self.raw_data, dim=-1)
        return self._processed_cache
    
    def invalidate_cache(self):
        """Clear cached computations when raw data changes."""
        self._processed_cache = None
    
    def update_raw_data(self, new_data: torch.Tensor):
        """Update raw data and invalidate dependent caches."""
        self.raw_data = new_data
        self.invalidate_cache()

# Usage
batch = LazyBatch(
    raw_data=torch.randn(32, 128),
    shape=(32,),
    device='cpu'
)

# First access computes and caches
processed1 = batch.processed_data  # Computation happens here
processed2 = batch.processed_data  # Returns cached result

# Update invalidates cache
batch.update_raw_data(torch.randn(32, 128))
processed3 = batch.processed_data  # Recomputes
```

### Batch Processing Utilities

```python
class ProcessingBatch(TensorDataClass):
    data: torch.Tensor
    
    def chunk(self, chunk_size: int) -> List['ProcessingBatch']:
        """Split batch into smaller chunks."""
        chunks = []
        for i in range(0, self.shape[0], chunk_size):
            end_idx = min(i + chunk_size, self.shape[0])
            chunks.append(self[i:end_idx])
        return chunks
    
    def shuffle(self, generator: Optional[torch.Generator] = None) -> 'ProcessingBatch':
        """Return shuffled version of batch."""
        indices = torch.randperm(self.shape[0], generator=generator)
        return self[indices]
    
    def sample(self, n: int, replace: bool = False) -> 'ProcessingBatch':
        """Sample n items from batch."""
        if replace:
            indices = torch.randint(0, self.shape[0], (n,))
        else:
            indices = torch.randperm(self.shape[0])[:n]
        return self[indices]
    
    def filter(self, mask: torch.Tensor) -> 'ProcessingBatch':
        """Filter batch using boolean mask."""
        return self[mask]

# Usage
large_batch = ProcessingBatch(
    data=torch.randn(1000, 128),
    shape=(1000,),
    device='cpu'
)

# Process in chunks to manage memory
for chunk in large_batch.chunk(100):
    process_chunk(chunk)

# Random sampling
sample_batch = large_batch.sample(50)

# Filtering
valid_mask = (large_batch.data.sum(dim=1) > 0)
filtered_batch = large_batch.filter(valid_mask)
```

## Best Practices

### Design Guidelines

1. **Field Organization**: Group related fields logically and use descriptive names
2. **Type Annotations**: Always provide explicit type annotations for better IDE support
3. **Default Values**: Use `field(default_factory=...)` for mutable defaults
4. **Validation**: Implement `__post_init__` for custom validation logic
5. **Documentation**: Add docstrings to classes and complex fields

### Performance Considerations

1. **Memory Sharing**: Be aware that operations create views, not copies
2. **Batch Operations**: Apply operations to entire TensorDataClass instances rather than individual fields
3. **Device Consistency**: Keep related tensors on the same device
4. **Caching**: Use computed properties for expensive derived values
5. **Serialization**: Consider separate tensor and metadata serialization for efficiency

### Integration Tips

1. **Model Compatibility**: Design batches to match your model's expected inputs
2. **DataLoader Integration**: Implement `__getitem__` for custom indexing if needed
3. **Metric Collection**: Include fields for tracking training metrics
4. **Debugging**: Add utility methods for inspection and validation
5. **Extensibility**: Use inheritance and mixins for reusable functionality

TensorDataClass excels in production environments where type safety, performance, and maintainability are crucial. Its static structure and comprehensive IDE support make it ideal for large codebases and team development scenarios.