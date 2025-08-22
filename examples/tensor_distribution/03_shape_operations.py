"""
TensorDistribution shape operations.

Key concept: Shape operations (reshape, view, squeeze, unsqueeze) work exactly
like tensor operations, applying only to batch dimensions while automatically
preserving each distribution's event dimensions. This makes sequence processing,
batch manipulation, and hierarchical modeling intuitive.

Key concepts demonstrated:
- Batch vs event dimension preservation  
- Reshape and view operations for sequence processing
- Squeeze/unsqueeze for dimension manipulation
- Common ML workflow patterns (flatten → process → reshape)
- Error handling for invalid transformations
"""

import torch
from tensorcontainer.tensor_distribution import TensorNormal


def demonstrate_batch_vs_event_dimensions():
    """Show how batch and event dimensions are handled differently."""
    print("=== Batch vs Event Dimensions ===")
    
    # Create distributions: 20 timesteps, 16 batch size, 5D actions
    sequence_dist = TensorNormal(
        loc=torch.randn(20, 16, 5),
        scale=torch.ones(20, 16, 5),
        shape=(20, 16),        # Batch shape: (timesteps, batch_size)
        device='cpu'
    )
    
    print(f"Original batch shape: {sequence_dist.batch_shape}")  # (20, 16)
    print(f"Original event shape: {sequence_dist.event_shape}")  # (5,)
    print(f"Sample shape: {sequence_dist.sample().shape}")       # (20, 16, 5)
    print()
    
    return sequence_dist


def demonstrate_sequence_processing_workflow():
    """Show common sequence processing patterns."""
    print("=== Sequence Processing Workflow ===")
    
    # Start with sequence: (timesteps=20, batch_size=16) 
    sequence_dist = TensorNormal(
        loc=torch.randn(20, 16, 5),
        scale=torch.ones(20, 16, 5),
        shape=(20, 16),
        device='cpu'
    )
    
    # Flatten sequence for batch processing
    flat_dist = sequence_dist.view(-1)     # Shape: (320,) = 20*16
    print(f"Flattened to: {flat_dist.shape} (20*16={flat_dist.shape[0]})")
    assert flat_dist.shape == (320,)
    assert flat_dist.sample().shape == (320, 5)  # Event dims preserved
    
    # Process in smaller chunks
    chunk_dist = flat_dist.view(64, 5)    # Shape: (64, 5) for processing
    print(f"Chunked to: {chunk_dist.shape}")
    assert chunk_dist.shape == (64, 5)
    assert chunk_dist.sample().shape == (64, 5, 5)  # Event dims still (5,)
    
    # Reshape back to sequence format
    back_to_sequence = chunk_dist.view(20, 16)
    assert back_to_sequence.shape == (20, 16)
    assert back_to_sequence.sample().shape == (20, 16, 5)
    
    print(f"Reshaped back to sequence: {back_to_sequence.shape}")
    print("Event dimensions (5,) preserved throughout all transformations")
    print()


def demonstrate_dimension_manipulation():
    """Show squeeze, unsqueeze, and expand operations."""
    print("=== Dimension Manipulation ===")
    
    # Start with simple batch
    dist = TensorNormal(
        loc=torch.randn(32, 4),
        scale=torch.ones(32, 4),
        shape=(32,),
        device='cpu'
    )
    
    # Add dimensions with unsqueeze
    unsqueezed = dist.unsqueeze(0)        # Add leading dim: (1, 32)
    print(f"Unsqueezed: {dist.shape} → {unsqueezed.shape}")
    assert unsqueezed.shape == (1, 32)
    assert unsqueezed.sample().shape == (1, 32, 4)
    
    # Add multiple dimensions
    multi_unsqueezed = dist.unsqueeze(0).unsqueeze(0)  # (1, 1, 32)
    print(f"Multi-unsqueezed: {multi_unsqueezed.shape}")
    assert multi_unsqueezed.shape == (1, 1, 32)
    
    # Remove size-1 dimensions with squeeze
    squeezed = multi_unsqueezed.squeeze()  # Back to (32,)
    print(f"Squeezed: {multi_unsqueezed.shape} → {squeezed.shape}")
    assert squeezed.shape == (32,)
    
    # Expand dimensions (doesn't copy data)
    expanded = dist.expand(2, -1)         # (2, 32) - broadcast along new dim
    print(f"Expanded: {dist.shape} → {expanded.shape}")
    assert expanded.shape == (2, 32)
    assert expanded.sample().shape == (2, 32, 4)
    print()


def demonstrate_hierarchical_modeling():
    """Show hierarchical modeling with shape operations.""" 
    print("=== Hierarchical Modeling ===")
    
    # Create hierarchical structure: 5 groups, 10 items per group
    group_means = torch.randn(5, 1, 8)     # Shared within each group
    individual_noise = torch.randn(5, 10, 8) * 0.1
    
    hierarchical_dist = TensorNormal(
        loc=group_means + individual_noise,
        scale=torch.ones(5, 10, 8),
        shape=(5, 10),                      # 5 groups, 10 items each
        device='cpu'
    )
    
    print(f"Hierarchical structure: {hierarchical_dist.shape}")
    print(f"Sample shape: {hierarchical_dist.sample().shape}")
    
    # Work with individual groups
    group_0 = hierarchical_dist[0]          # Just group 0: shape (10,)
    print(f"Single group: {group_0.shape}")
    assert group_0.shape == (10,)
    
    # Flatten hierarchy for processing
    flat_items = hierarchical_dist.view(-1) # All items: shape (50,)
    print(f"Flattened hierarchy: {flat_items.shape}")
    assert flat_items.shape == (50,)
    assert flat_items.sample().shape == (50, 8)
    
    # Reshape for different groupings
    different_grouping = flat_items.view(10, 5)  # 10 groups of 5
    print(f"Regrouped: {different_grouping.shape}")
    assert different_grouping.shape == (10, 5)
    print()


def demonstrate_error_handling():
    """Show validation and error handling for invalid operations."""
    print("=== Shape Operation Validation ===")
    
    dist = TensorNormal(
        loc=torch.randn(6, 4),
        scale=torch.ones(6, 4),
        shape=(6,),
        device='cpu'
    )
    
    # Invalid reshape - incompatible total elements
    try:
        dist.reshape(5)  # Cannot reshape (6,) to (5,)
        assert False, "Should have raised an error"
    except Exception as e:
        print(f"Invalid reshape caught: {type(e).__name__}")
    
    # Valid reshapes
    reshaped_2x3 = dist.reshape(2, 3)      # (6,) → (2, 3)
    assert reshaped_2x3.shape == (2, 3)
    print(f"Valid reshape: (6,) → {reshaped_2x3.shape}")
    
    reshaped_1x6 = dist.reshape(1, 6)      # (6,) → (1, 6) 
    assert reshaped_1x6.shape == (1, 6)
    print(f"Valid reshape: (6,) → {reshaped_1x6.shape}")
    
    # Event dimensions always preserved
    assert reshaped_2x3.sample().shape == (2, 3, 4)
    assert reshaped_1x6.sample().shape == (1, 6, 4)
    print("Event dimensions (4,) preserved in all reshapes")


def main() -> None:
    """Demonstrate shape operations with TensorDistribution."""
    
    demonstrate_batch_vs_event_dimensions()
    demonstrate_sequence_processing_workflow() 
    demonstrate_dimension_manipulation()
    demonstrate_hierarchical_modeling()
    demonstrate_error_handling()
    
    print("\n=== Key Insights ===")
    print("1. Shape operations work exactly like tensor operations")
    print("2. Batch dimensions are transformed, event dimensions preserved") 
    print("3. No manual parameter reshaping needed (unlike torch.distributions)")
    print("4. Enables natural ML workflows: sequences, hierarchies, batching")
    print("5. Shape validation prevents common errors")


if __name__ == "__main__":
    main()