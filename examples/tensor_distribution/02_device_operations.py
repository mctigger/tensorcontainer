"""
TensorDistribution device operations.

Key concept: Moving distributions between devices (CPU ↔ GPU) is effortless
with TensorDistribution. A single `.to()` call handles all parameters automatically,
eliminating the manual parameter tracking required by torch.distributions.

Key concepts demonstrated:
- Unified device management with .to() method
- Automatic parameter synchronization across devices
- Device consistency validation
- Comparison with torch.distributions manual approach
- Mixed device scenarios
"""

import torch
from torch.distributions import Normal
from tensorcontainer.tensor_distribution import TensorNormal


def demonstrate_torch_distributions_pain():
    """Show the manual work required with torch.distributions."""
    print("=== torch.distributions approach (manual) ===")
    
    # Create distributions on CPU
    loc_cpu = torch.zeros(1000, 10)
    scale_cpu = torch.ones(1000, 10)
    dist_cpu = Normal(loc_cpu, scale_cpu)
    
    print(f"Original device: {loc_cpu.device}")
    
    # Moving to GPU requires manual reconstruction
    loc_gpu = loc_cpu.to('cuda')
    scale_gpu = scale_cpu.to('cuda')
    dist_gpu = Normal(loc_gpu, scale_gpu)
    
    # You have to manually track and move every parameter
    # If you forget one parameter, you get device mismatch errors
    print(f"After manual move: {loc_gpu.device}")
    print("Required manual parameter tracking and reconstruction")
    print()


def demonstrate_tensor_distribution_ease():
    """Show the elegance of TensorDistribution device management."""
    print("=== TensorDistribution approach (automatic) ===")
    
    # Create distribution on CPU
    dist_cpu = TensorNormal(
        loc=torch.zeros(1000, 10),
        scale=torch.ones(1000, 10)
    )
    
    print(f"Original device: {dist_cpu.device}")
    
    # Move to GPU with a single operation
    dist_gpu = dist_cpu.to('cuda')
    
    # All parameters automatically moved and distribution reconstructed
    print(f"After .to('cuda'): {dist_gpu.device}")
    print("Automatic parameter management - no manual tracking needed!")
    print()
    
    return dist_cpu, dist_gpu


def main() -> None:
    """Demonstrate device operations with TensorDistribution."""
    
    # Show the contrast between manual and automatic approaches
    demonstrate_torch_distributions_pain()
    dist_cpu, dist_gpu = demonstrate_tensor_distribution_ease()
    
    # Verify device consistency
    assert dist_gpu.device == torch.device('cuda:0')
    assert dist_gpu._loc.device == torch.device('cuda:0')
    assert dist_gpu._scale.device == torch.device('cuda:0')
    
    # Operations work seamlessly on any device
    cpu_samples = dist_cpu.sample()     # CPU computation
    gpu_samples = dist_gpu.sample()     # GPU computation
    
    assert cpu_samples.device == torch.device('cpu')
    assert gpu_samples.device == torch.device('cuda:0')
    
    # Move back to CPU
    dist_back_to_cpu = dist_gpu.to('cpu')
    assert dist_back_to_cpu.device == torch.device('cpu')
    
    # Complex distributions with multiple parameters work the same way
    complex_dist = TensorNormal(
        loc=torch.randn(50, 20, requires_grad=True),
        scale=torch.ones(50, 20, requires_grad=True) * 0.5,
        shape=(50,),
        device='cpu'
    )
    
    # Even with gradients, device movement is seamless
    complex_gpu = complex_dist.to('cuda')
    assert complex_gpu._loc.requires_grad
    assert complex_gpu._scale.requires_grad
    assert complex_gpu._loc.device == torch.device('cuda:0')
    
    # Test mixed device scenarios (advanced use case)
    # Setting device=None allows tensors on different devices
    mixed_device_dist = complex_gpu.to(device=None)
    assert mixed_device_dist.device is None
    
    # This becomes crucial in distributed training scenarios
    print("=== Advanced: Complex distribution device movement ===")
    print(f"Complex distribution moved from CPU to GPU: {complex_gpu.device}")
    print(f"Parameters retain gradients: loc={complex_gpu._loc.requires_grad}, scale={complex_gpu._scale.requires_grad}")
    print(f"Mixed device mode: {mixed_device_dist.device}")
    
    # Demonstrate training scenario: move batch to GPU for processing
    print("\n=== Training scenario: Batch processing ===")
    
    # Simulate policy network output (batch of distributions)
    policy_distributions = TensorNormal(
        loc=torch.randn(256, 6),        # 256 states, 6D actions
        scale=torch.ones(256, 6) * 0.2,
        shape=(256,),
        device='cpu'
    )
    
    # Move entire batch to GPU for processing
    gpu_policy = policy_distributions.to('cuda')
    
    # Sample actions on GPU
    gpu_actions = gpu_policy.sample()
    assert gpu_actions.device == torch.device('cuda:0')
    assert gpu_actions.shape == (256, 6)
    
    print(f"Moved batch of {gpu_policy.shape[0]} distributions to GPU")
    print(f"Sampled actions shape: {gpu_actions.shape} on {gpu_actions.device}")
    
    # The key insight: regardless of complexity, it's always just .to(device)
    # No parameter tracking, no manual reconstruction, no device mismatch errors


if __name__ == "__main__":
    main()