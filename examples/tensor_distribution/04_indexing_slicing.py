"""
TensorDistribution indexing and slicing.

Key concept: TensorDistribution supports all tensor indexing patterns - 
boolean masking, advanced indexing, slicing - making it natural to select
subsets of distributions based on conditions. This is essential for filtering,
conditional processing, and dynamic batch management in ML workflows.

Key concepts demonstrated:
- Boolean masking for conditional distribution selection
- Advanced indexing for specific distribution selection  
- Slicing patterns for batch processing
- Dynamic filtering based on runtime conditions
- Integration with tensor operations for seamless workflows
"""

import torch
from tensorcontainer.tensor_distribution import TensorNormal


def demonstrate_basic_indexing():
    """Show basic indexing operations."""
    print("=== Basic Indexing Operations ===")
    
    # Create batch of distributions for different environments
    env_dists = TensorNormal(
        loc=torch.randn(100, 4),            # 100 environments, 4D actions
        scale=torch.ones(100, 4),
        shape=(100,),
        device='cpu'
    )
    
    # Single distribution
    single_dist = env_dists[0]              # First environment's distribution
    print(f"Single distribution: {env_dists.shape} → {single_dist.shape}")
    assert single_dist.shape == ()          # Scalar batch shape
    assert single_dist.sample().shape == (4,)  # Still 4D actions
    
    # Slice ranges
    first_ten = env_dists[:10]              # First 10 environments
    print(f"First 10: {first_ten.shape}")
    assert first_ten.shape == (10,)
    assert first_ten.sample().shape == (10, 4)
    
    # Strided slicing
    every_fifth = env_dists[::5]            # Every 5th environment
    print(f"Every 5th: {every_fifth.shape}")
    assert every_fifth.shape == (20,)       # 100/5 = 20
    
    # Negative indexing
    last_dist = env_dists[-1]               # Last environment
    last_five = env_dists[-5:]              # Last 5 environments
    print(f"Last distribution: {last_dist.shape}")
    print(f"Last 5: {last_five.shape}")
    assert last_five.shape == (5,)
    print()


def demonstrate_boolean_masking():
    """Show boolean masking for conditional selection."""
    print("=== Boolean Masking for Conditional Selection ===")
    
    # Create distributions with varying performance scores
    performance_scores = torch.rand(100)    # Random performance scores [0, 1]
    
    env_dists = TensorNormal(
        loc=torch.randn(100, 4),
        scale=torch.ones(100, 4) * 0.5,
        shape=(100,),
        device='cpu'
    )
    
    # Select high-performing environments (score > 0.8)
    high_performance_mask = performance_scores > 0.8
    high_perf_dists = env_dists[high_performance_mask]
    
    print(f"High performance environments: {high_perf_dists.shape[0]}/{len(env_dists)}")
    print(f"Selection percentage: {high_perf_dists.shape[0]/len(env_dists)*100:.1f}%")
    
    # Sample only from high-performing environments
    high_perf_actions = high_perf_dists.sample()
    print(f"High-performance actions shape: {high_perf_actions.shape}")
    
    # Multiple conditions
    medium_to_high = (performance_scores > 0.5) & (performance_scores < 0.9)
    medium_high_dists = env_dists[medium_to_high]
    print(f"Medium-high performance: {medium_high_dists.shape[0]} environments")
    
    # Dynamic filtering during training
    # Example: Only train on environments that aren't solved yet
    solved_threshold = 0.95
    unsolved_mask = performance_scores < solved_threshold
    training_dists = env_dists[unsolved_mask]
    
    print(f"Unsolved environments for training: {training_dists.shape[0]}")
    print()


def demonstrate_advanced_indexing():
    """Show advanced indexing with specific selections."""
    print("=== Advanced Indexing for Specific Selection ===")
    
    env_dists = TensorNormal(
        loc=torch.randn(100, 6),            # 100 environments, 6D actions
        scale=torch.ones(100, 6) * 0.3,
        shape=(100,),
        device='cpu'
    )
    
    # Select specific environments by index
    interesting_envs = torch.tensor([5, 12, 23, 47, 81, 99])
    selected_dists = env_dists[interesting_envs]
    
    print(f"Selected specific environments: {selected_dists.shape}")
    assert selected_dists.shape == (6,)     # 6 selected environments
    assert selected_dists.sample().shape == (6, 6)
    
    # Random sampling of environments
    random_indices = torch.randperm(100)[:20]  # Random 20 environments
    random_dists = env_dists[random_indices]
    print(f"Random sample: {random_dists.shape}")
    
    # Top-K selection based on some criterion
    values = torch.randn(100)               # Some metric to sort by
    top_k_indices = torch.topk(values, k=15).indices
    top_k_dists = env_dists[top_k_indices]
    print(f"Top-15 environments: {top_k_dists.shape}")
    
    # Fancy indexing with multiple dimensions
    # Note: This creates a copy, not a view
    grid_indices = torch.tensor([[0, 1, 2], [10, 11, 12]])
    grid_selection = env_dists[grid_indices.flatten()]
    print(f"Grid selection: {grid_selection.shape}")
    print()


def demonstrate_multidimensional_indexing():
    """Show indexing with multidimensional batch shapes."""
    print("=== Multidimensional Batch Indexing ===")
    
    # Create hierarchical distributions: 10 groups, 8 items per group
    hierarchical_dists = TensorNormal(
        loc=torch.randn(10, 8, 5),
        scale=torch.ones(10, 8, 5),
        shape=(10, 8),                      # 2D batch shape
        device='cpu'
    )
    
    # Index first dimension (select groups)
    group_3 = hierarchical_dists[3]         # Group 3: shape (8,)
    print(f"Single group: {hierarchical_dists.shape} → {group_3.shape}")
    assert group_3.shape == (8,)
    
    # Index both dimensions (select specific item)
    item_3_5 = hierarchical_dists[3, 5]     # Group 3, Item 5: shape ()
    print(f"Specific item: {item_3_5.shape}")
    assert item_3_5.shape == ()
    assert item_3_5.sample().shape == (5,)
    
    # Slice in multiple dimensions
    subset_groups = hierarchical_dists[:3, 2:6]  # First 3 groups, items 2-5
    print(f"Subset: {subset_groups.shape}")
    assert subset_groups.shape == (3, 4)
    
    # Boolean mask on multidimensional
    group_quality = torch.rand(10) > 0.7    # Quality mask for groups
    good_groups = hierarchical_dists[group_quality]
    print(f"High-quality groups: {good_groups.shape}")
    print()


def demonstrate_dynamic_batch_management():
    """Show dynamic batch management scenarios."""
    print("=== Dynamic Batch Management ===")
    
    # Simulate training scenario with variable batch sizes
    policy_dists = TensorNormal(
        loc=torch.randn(256, 4),            # Full batch of policy outputs
        scale=torch.ones(256, 4) * 0.2,
        shape=(256,),
        device='cpu'
    )
    
    # Process in chunks (memory constraints)
    chunk_size = 64
    processed_chunks = []
    
    for i in range(0, len(policy_dists), chunk_size):
        chunk = policy_dists[i:i+chunk_size]
        # Simulate processing (e.g., GPU memory constraints)
        chunk_actions = chunk.sample()
        processed_chunks.append(chunk_actions)
        print(f"Processed chunk {i//chunk_size + 1}: {chunk.shape}")
    
    # Combine results
    all_actions = torch.cat(processed_chunks, dim=0)
    assert all_actions.shape == (256, 4)
    
    # Dynamic filtering during training
    # Remove distributions that produced invalid actions
    valid_actions_mask = torch.all(torch.abs(all_actions) < 10.0, dim=1)
    valid_policy_dists = policy_dists[valid_actions_mask]
    
    print(f"Valid policies after filtering: {valid_policy_dists.shape[0]}/{len(policy_dists)}")
    
    # Adaptive batch sizing based on performance
    performance_metric = torch.rand(len(valid_policy_dists))
    
    # High performers get more samples
    high_perf_mask = performance_metric > 0.8
    low_perf_mask = performance_metric <= 0.3
    
    high_perf_dists = valid_policy_dists[high_perf_mask]
    low_perf_dists = valid_policy_dists[low_perf_mask]
    
    if len(high_perf_dists) > 0:
        high_perf_samples = high_perf_dists.sample((5,))  # 5 samples each
        print(f"High performers: {len(high_perf_dists)} dists, {high_perf_samples.shape} samples")
    
    if len(low_perf_dists) > 0:
        low_perf_samples = low_perf_dists.sample()        # 1 sample each
        print(f"Low performers: {len(low_perf_dists)} dists, {low_perf_samples.shape} samples")
    print()


def demonstrate_integration_with_tensor_ops():
    """Show integration with other tensor operations."""
    print("=== Integration with Tensor Operations ===")
    
    # Start with batch of distributions
    dists = TensorNormal(
        loc=torch.randn(50, 3),
        scale=torch.ones(50, 3),
        shape=(50,),
        device='cpu'
    )
    
    # Combine indexing with other tensor operations
    # Select subset and move to GPU
    subset = dists[:20].to('cuda')
    print(f"Selected and moved to GPU: {subset.shape} on {subset.device}")
    
    # Index, reshape, then sample
    selected = dists[torch.tensor([0, 5, 10, 15, 20, 25])]
    reshaped = selected.view(2, 3)          # Reshape to 2x3
    samples = reshaped.sample()
    
    print(f"Index → reshape → sample: {selected.shape} → {reshaped.shape} → {samples.shape}")
    assert samples.shape == (2, 3, 3)      # (2, 3) batch + (3,) event
    
    # Chain multiple operations
    result = dists[10:40].view(5, 6).unsqueeze(0)[0, :3]
    print(f"Chained operations result: {result.shape}")
    assert result.shape == (3, 6)


def main() -> None:
    """Demonstrate indexing and slicing with TensorDistribution."""
    
    demonstrate_basic_indexing()
    demonstrate_boolean_masking()
    demonstrate_advanced_indexing()
    demonstrate_multidimensional_indexing()
    demonstrate_dynamic_batch_management()
    demonstrate_integration_with_tensor_ops()
    
    print("=== Key Insights ===")
    print("1. All tensor indexing patterns work with TensorDistribution")
    print("2. Boolean masking enables conditional distribution selection")
    print("3. Advanced indexing supports complex selection patterns")
    print("4. Perfect integration with other tensor operations")
    print("5. Enables dynamic batch management and filtering workflows")
    print("6. Essential for adaptive training and inference scenarios")


if __name__ == "__main__":
    main()