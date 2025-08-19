# TensorDict Deep Dive

TensorDict provides a dictionary-like interface for managing collections of tensors with shared batch dimensions. This guide covers advanced usage patterns, nested structures, and specialized operations that make TensorDict ideal for dynamic data scenarios.

## Creation and Initialization

### Basic Construction

```python
import torch
from tensorcontainer import TensorDict

# Simple key-value construction
data = TensorDict({
    'observations': torch.randn(32, 128),
    'actions': torch.randint(0, 4, (32,)),
    'rewards': torch.randn(32, 1)
}, shape=(32,), device='cpu')
```

### From Existing Data Structures

```python
# Convert from regular dictionary
raw_data = {
    'states': torch.randn(16, 64),
    'next_states': torch.randn(16, 64),
    'actions': torch.randn(16, 4)
}
td = TensorDict(raw_data, shape=(16,), device='cpu')

# Empty initialization for dynamic filling
empty_td = TensorDict({}, shape=(32,), device='cuda')
empty_td['observations'] = torch.randn(32, 128, device='cuda')
empty_td['actions'] = torch.randn(32, 4, device='cuda')
```

## Dictionary Operations

### Key Management

```python
# Check if key exists
if 'rewards' in data:
    rewards = data['rewards']

# Get all keys
all_keys = list(data.keys())
print(f"Available keys: {all_keys}")

# Get values and items
values = list(data.values())
items = list(data.items())

# Length returns number of top-level keys
print(f"Number of fields: {len(data)}")
```

### Dynamic Field Assignment

```python
# Add new fields dynamically
data['next_observations'] = torch.randn(32, 128)
data['done'] = torch.randint(0, 2, (32, 1))

# Remove fields
del data['rewards']

# Update multiple fields
data.update({
    'values': torch.randn(32, 1),
    'log_probs': torch.randn(32, 4)
})
```

### Safe Access Patterns

```python
# Get with default value
rewards = data.get('rewards', torch.zeros(32, 1))

# Pop with default
old_actions = data.pop('old_actions', torch.zeros(32, 4))

# Set default if key doesn't exist
data.setdefault('episode_id', torch.arange(32))
```

## Nested Structures

### Creating Nested TensorDicts

```python
# Nested dictionary construction
nested_data = TensorDict({
    'agent': {
        'position': torch.randn(32, 3),
        'velocity': torch.randn(32, 3),
        'health': torch.randint(0, 100, (32, 1))
    },
    'environment': {
        'obstacles': torch.randn(32, 10, 2),
        'goals': torch.randn(32, 2),
        'lighting': torch.randn(32, 1)
    },
    'game_state': {
        'score': torch.randint(0, 1000, (32, 1)),
        'level': torch.randint(1, 10, (32, 1))
    }
}, shape=(32,), device='cpu')

# Access nested values
agent_pos = nested_data['agent']['position']
obstacle_count = nested_data['environment']['obstacles'].shape[1]  # 10
```

### Deep Nesting Operations

```python
# Create deeply nested structure
deep_nested = TensorDict({
    'level1': {
        'level2': {
            'level3': {
                'data': torch.randn(32, 64)
            }
        }
    }
}, shape=(32,), device='cpu')

# Access deep values
deep_data = deep_nested['level1']['level2']['level3']['data']

# Add to deep structure
deep_nested['level1']['level2']['more_data'] = torch.randn(32, 32)
```

### Nested Batch Operations

```python
# All nested tensors are affected by batch operations
reshaped_nested = nested_data.reshape(8, 4)
# agent.position becomes (8, 4, 3)
# environment.obstacles becomes (8, 4, 10, 2)

moved_nested = nested_data.to('cuda')
# All nested tensors moved to CUDA

indexed_nested = nested_data[0]
# All nested tensors indexed, batch shape becomes ()
```

## Key Flattening

### Basic Flattening

```python
# Flatten nested keys with dot notation
flat_data = nested_data.flatten_keys()
print(list(flat_data.keys()))
# ['agent.position', 'agent.velocity', 'agent.health', 
#  'environment.obstacles', 'environment.goals', 'environment.lighting',
#  'game_state.score', 'game_state.level']

# Access flattened keys
agent_pos = flat_data['agent.position']
env_goals = flat_data['environment.goals']
```

### Custom Separators

```python
# Use custom separator
flat_custom = nested_data.flatten_keys(separator='/')
print(list(flat_custom.keys()))
# ['agent/position', 'agent/velocity', 'agent/health', ...]

# Or use underscore
flat_underscore = nested_data.flatten_keys(separator='_')
print(list(flat_underscore.keys()))
# ['agent_position', 'agent_velocity', 'agent_health', ...]
```

### Working with Flattened Data

```python
# Flattened data is still a TensorDict
assert isinstance(flat_data, TensorDict)

# All tensor operations work normally
flattened_reshaped = flat_data.reshape(8, 4)
flattened_cuda = flat_data.to('cuda')

# Can still add new flattened keys
flat_data['new.deeply.nested.value'] = torch.randn(32, 16)
```

## Advanced Indexing and Slicing

### Boolean Masking

```python
# Create boolean mask
active_mask = torch.randint(0, 2, (32,)).bool()

# Apply mask to entire TensorDict
active_data = data[active_mask]
print(f"Active data shape: {active_data.shape}")  # Shape varies based on mask

# Works with nested structures
active_nested = nested_data[active_mask]
# All nested tensors are masked consistently
```

### Advanced Indexing Patterns

```python
# Multi-dimensional indexing (for higher-dimensional batch shapes)
time_series_data = TensorDict({
    'states': torch.randn(8, 10, 64),    # (batch, time, features)
    'actions': torch.randn(8, 10, 4),
    'rewards': torch.randn(8, 10, 1)
}, shape=(8, 10), device='cpu')

# Index specific episodes and timesteps
episode_subset = time_series_data[:4, :5]  # First 4 episodes, first 5 timesteps
last_timestep = time_series_data[:, -1]    # All episodes, last timestep
middle_episodes = time_series_data[2:6, :] # Episodes 2-5, all timesteps
```

### Ellipsis and Complex Indexing

```python
# Use ellipsis for complex indexing
complex_data = TensorDict({
    'multi_dim': torch.randn(4, 8, 6, 128),
    'simple': torch.randn(4, 8, 6, 1)
}, shape=(4, 8, 6), device='cpu')

# Ellipsis indexing
subset = complex_data[..., :3]  # Last batch dimension sliced
middle = complex_data[:, ..., 2:4]  # Middle slice of second-to-last dimension
```

## Memory and Performance Optimization

### Sharing Memory

```python
# TensorDict shares memory with original tensors
original_tensor = torch.randn(32, 128)
data = TensorDict({'obs': original_tensor}, shape=(32,), device='cpu')

# Modifying original affects TensorDict
original_tensor[0] = 999
assert data['obs'][0, 0] == 999

# Views also share memory
reshaped_data = data.reshape(8, 4)
reshaped_data['obs'][0, 0, 0] = 777
assert data['obs'][0, 0] == 777
```

### Efficient Updates

```python
# Batch updates are more efficient than individual assignments
updates = {
    'new_obs': torch.randn(32, 128),
    'new_actions': torch.randn(32, 4),
    'new_rewards': torch.randn(32, 1)
}
data.update(updates)  # Single operation

# Avoid inefficient patterns
# BAD: Multiple individual assignments in loop
# for key, value in updates.items():
#     data[key] = value  # Less efficient
```

### Memory-Conscious Construction

```python
# Use unsafe_construction for performance-critical paths
from tensorcontainer import TensorContainer

with TensorContainer.unsafe_construction():
    batches = []
    for batch_idx in range(1000):
        batch_data = TensorDict({
            'obs': torch.randn(32, 128),
            'actions': torch.randn(32, 4)
        }, shape=(32,), device='cpu')
        batches.append(batch_data)
```

## Serialization and Persistence

### Dictionary Conversion

```python
# Convert to regular Python dictionary
python_dict = dict(data.items())
nested_python_dict = {k: v.cpu().numpy() for k, v in data.items()}

# Reconstruct from dictionary
reconstructed = TensorDict(python_dict, shape=data.shape, device=data.device)
```

### State Management

```python
# Extract state for serialization
state = {
    'data': {k: v.cpu() for k, v in data.items()},
    'shape': data.shape,
    'device': str(data.device)
}

# Restore from state
restored_data = TensorDict(
    state['data'], 
    shape=state['shape'], 
    device=state['device']
)
```

## Integration Patterns

### PyTree Operations

```python
import torch.utils._pytree as pytree

# Apply functions to all tensors
def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)

normalized_data = pytree.tree_map(normalize, data)

# Combine multiple TensorDicts
def weighted_average(x, y, weight=0.5):
    return weight * x + (1 - weight) * y

averaged = pytree.tree_map(weighted_average, data1, data2)
```

### DataLoader Integration

```python
from torch.utils.data import DataLoader, Dataset

class TensorDictDataset(Dataset):
    def __init__(self, tensor_dict):
        self.data = tensor_dict
        self.length = tensor_dict.shape[0]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.data[idx]

# Create dataset and dataloader
dataset = TensorDictDataset(data)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

for batch in dataloader:
    # Each batch is a TensorDict with batch_size samples
    assert isinstance(batch, TensorDict)
    process_batch(batch)
```

### Model Integration

```python
import torch.nn as nn

class TensorDictProcessor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_encoder = nn.Linear(obs_dim, 64)
        self.action_encoder = nn.Linear(action_dim, 32)
        self.combiner = nn.Linear(96, 1)
    
    def forward(self, data):
        # Process TensorDict directly
        obs_features = self.obs_encoder(data['observations'])
        action_features = self.action_encoder(data['actions'])
        combined = torch.cat([obs_features, action_features], dim=-1)
        return self.combiner(combined)

# Model can process TensorDict directly
model = TensorDictProcessor(128, 4)
output = model(data)
```

## Error Handling and Debugging

### Common Error Patterns

```python
# Shape mismatch errors provide detailed paths
try:
    invalid_data = TensorDict({
        'good_tensor': torch.randn(32, 64),
        'bad_tensor': torch.randn(16, 64),  # Wrong batch size
        'nested': {
            'also_bad': torch.randn(8, 32)   # Also wrong batch size
        }
    }, shape=(32,), device='cpu')
except ValueError as e:
    print(f"Detailed error: {e}")
    # Shows exact path to problematic tensor
```

### Validation Utilities

```python
# Check shape consistency
def validate_tensordict(td):
    for key, value in td.items():
        if isinstance(value, torch.Tensor):
            expected_prefix = td.shape
            actual_prefix = value.shape[:len(expected_prefix)]
            if actual_prefix != expected_prefix:
                print(f"Shape mismatch at {key}: expected {expected_prefix}, got {actual_prefix}")

# Check device consistency
def validate_devices(td):
    if td.device is not None:
        for key, value in td.items():
            if isinstance(value, torch.Tensor) and value.device != td.device:
                print(f"Device mismatch at {key}: expected {td.device}, got {value.device}")
```

## Best Practices

### Naming Conventions

```python
# Use descriptive, hierarchical keys
good_names = TensorDict({
    'agent.observations.visual': torch.randn(32, 3, 84, 84),
    'agent.observations.proprioception': torch.randn(32, 12),
    'agent.actions.discrete': torch.randint(0, 4, (32,)),
    'agent.actions.continuous': torch.randn(32, 2),
    'environment.rewards.dense': torch.randn(32, 1),
    'environment.rewards.sparse': torch.randn(32, 1)
}, shape=(32,), device='cpu')
```

### Structure Design

```python
# Group related data logically
well_structured = TensorDict({
    'inputs': {
        'observations': torch.randn(32, 128),
        'previous_actions': torch.randn(32, 4)
    },
    'targets': {
        'actions': torch.randn(32, 4),
        'values': torch.randn(32, 1)
    },
    'metadata': {
        'episode_ids': torch.arange(32),
        'timesteps': torch.randint(0, 1000, (32,))
    }
}, shape=(32,), device='cpu')
```

### Performance Guidelines

1. **Use batch operations**: Apply operations to entire TensorDict rather than individual tensors
2. **Minimize key changes**: Avoid frequently adding/removing keys in performance-critical code
3. **Consider flattening**: For ML pipelines, flattened keys often provide better performance
4. **Memory sharing**: Be aware of when tensors share memory vs. when they're copied
5. **Device consistency**: Keep related tensors on the same device when possible

TensorDict excels in scenarios requiring flexible, dynamic data structures with the safety and performance benefits of the TensorContainer framework. Its dictionary-like interface makes it intuitive for developers familiar with Python dictionaries while providing the advanced tensor management capabilities needed for complex ML workflows.
