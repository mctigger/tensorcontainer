# TensorDistribution Examples

This directory contains educational examples demonstrating TensorDistribution usage patterns. Each example focuses on specific concepts and builds upon previous ones.

## Core Examples

### 01_basic.py - First Steps
**Key Concept:** Distributions that behave like tensors

Demonstrates:
- Creating batches of distributions as single objects  
- Basic sampling operations (sample vs rsample)
- Probability computation and distribution properties
- Shape validation and error handling
- The fundamental advantage over torch.distributions

### 02_device_operations.py - Device Management  
**Key Concept:** Effortless GPU/CPU movement

Demonstrates:
- Moving distributions between devices with `.to()`
- Automatic parameter synchronization
- Comparison with manual torch.distributions approach
- Complex scenarios with gradients and mixed devices

### 03_shape_operations.py - Shape Transformations
**Key Concept:** Tensor-like shape operations  

Demonstrates:
- Reshape, view, squeeze, unsqueeze operations
- Batch dimension transformations
- Sequence processing workflows  
- Hierarchical modeling patterns

### 04_indexing_slicing.py - Selecting Distributions
**Key Concept:** Intuitive distribution selection

Demonstrates:
- Boolean masking for conditional selection
- Advanced indexing patterns
- Dynamic batch management
- Integration with tensor operations

### 05_stacking_concatenation.py - Combining Distributions  
**Key Concept:** Composable distribution operations

Demonstrates:
- torch.stack and torch.cat operations
- Multi-agent and ensemble scenarios
- Hierarchical composition patterns
- Policy comparison workflows

## Advanced Examples

### 06_sampling_patterns.py - Advanced Sampling Strategies
**Key Concept:** Different sampling approaches for various scenarios

Demonstrates:
- sample() vs rsample() for gradient flow
- Multiple sampling shapes and batch processing
- Custom sampling functions and strategies
- Statistical property computation from samples

### 07_gradient_flow.py - Training Integration
**Key Concept:** Seamless training with distributions

Demonstrates:
- Reparameterized sampling for VAE-style training
- Gradient tracking and detachment patterns
- Loss computation with distribution parameters
- Integration with PyTorch optimizers

### 08_integration_tensorcontainer.py - Ecosystem Integration
**Key Concept:** Working with TensorDict and TensorDataClass

Demonstrates:
- Nested distribution structures
- Unified operations on multiple distributions
- Type-safe distribution containers
- Complex hierarchical modeling

### 09_kl_divergence.py - Distribution Comparisons
**Key Concept:** Measuring distribution differences

Demonstrates:
- KL divergence between TensorDistributions
- Mixed TensorDistribution/torch.distributions operations
- Policy comparison and regularization workflows
- Interoperability with PyTorch's KL registry

### 10_real_world_workflows.py - Practical Applications
**Key Concept:** End-to-end usage patterns

Demonstrates:
- Policy networks outputting distributions
- Variational inference and Bayesian modeling
- Multi-task and multi-agent scenarios
- Production-ready patterns and best practices

## Running Examples

```bash
# Run individual examples
python examples/tensor_distribution/01_basic.py
python examples/tensor_distribution/02_device_operations.py

# Run all examples  
for f in examples/tensor_distribution/[0-9]*.py; do python "$f"; done
```

## Key Insights

1. **Unified Interface**: TensorDistribution makes probability distributions behave like tensors
2. **Automatic Management**: Device, shape, and parameter handling is automatic
3. **Natural Integration**: Works seamlessly with existing PyTorch tensor operations  
4. **Batch Operations**: Enables efficient batch processing of distribution operations
5. **Composability**: Supports complex hierarchical and multi-agent scenarios

## Design Philosophy  

These examples follow a progression:
1. **Core concept** - distributions as tensors
2. **Essential operations** - device, shape, indexing  
3. **Composition** - combining distributions
4. **Advanced usage** - real-world patterns
5. **Integration** - ecosystem compatibility

Each example is self-contained with clear explanations, but they build conceptually to demonstrate the full power of the TensorDistribution system.