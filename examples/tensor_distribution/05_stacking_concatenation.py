"""
TensorDistribution stacking and concatenation.

Key concept: TensorDistribution supports torch.stack and torch.cat operations,
making it natural to combine distributions from different sources. This enables
policy comparison, ensemble methods, multi-agent scenarios, and hierarchical
modeling patterns that are essential in modern ML workflows.

Key concepts demonstrated:
- torch.stack for creating new batch dimensions
- torch.cat for extending existing dimensions
- Policy comparison and ensemble scenarios
- Multi-agent and multi-task combination patterns
- Hierarchical composition of distribution collections
"""

import torch
from tensorcontainer.tensor_distribution import TensorNormal, TensorCategorical


def demonstrate_basic_stacking():
    """Show basic stacking operations."""
    print("=== Basic Stacking Operations ===")
    
    # Create two policy networks' outputs
    policy_a = TensorNormal(
        loc=torch.randn(32, 4),              # Policy A: 32 states, 4D actions
        scale=torch.ones(32, 4) * 0.5,
        shape=(32,),
        device='cpu'
    )
    
    policy_b = TensorNormal(
        loc=torch.randn(32, 4),              # Policy B: same structure
        scale=torch.ones(32, 4) * 0.3,      # Different exploration noise
        shape=(32,),
        device='cpu'
    )
    
    # Stack creates new dimension for comparison
    compared_policies = torch.stack([policy_a, policy_b], dim=0)
    print(f"Individual policies: {policy_a.shape}")
    print(f"Stacked policies: {compared_policies.shape}")
    assert compared_policies.shape == (2, 32)        # 2 policies, 32 states each
    
    # Sample from both policies simultaneously
    policy_samples = compared_policies.sample()      # Shape: (2, 32, 4)
    print(f"Policy samples shape: {policy_samples.shape}")
    assert policy_samples.shape == (2, 32, 4)
    
    # Access individual policies
    policy_a_reconstructed = compared_policies[0]
    policy_b_reconstructed = compared_policies[1]
    assert policy_a_reconstructed.shape == (32,)
    assert policy_b_reconstructed.shape == (32,)
    
    print("Stacking enables side-by-side policy comparison")
    print()


def demonstrate_concatenation():
    """Show concatenation operations for batch extension."""
    print("=== Concatenation for Batch Extension ===")
    
    # Create two batches of distributions
    batch_1 = TensorNormal(
        loc=torch.randn(16, 6),              # First batch: 16 samples
        scale=torch.ones(16, 6),
        shape=(16,),
        device='cpu'
    )
    
    batch_2 = TensorNormal(
        loc=torch.randn(24, 6),              # Second batch: 24 samples  
        scale=torch.ones(24, 6),
        shape=(24,),
        device='cpu'
    )
    
    # Concatenate to create larger batch
    combined_batch = torch.cat([batch_1, batch_2], dim=0)
    print(f"Batch 1: {batch_1.shape}, Batch 2: {batch_2.shape}")
    print(f"Combined: {combined_batch.shape}")
    assert combined_batch.shape == (40,)             # 16 + 24 = 40
    
    # Sample from combined batch
    combined_samples = combined_batch.sample()
    assert combined_samples.shape == (40, 6)
    print(f"Combined samples: {combined_samples.shape}")
    
    # Multiple concatenations
    batch_3 = TensorNormal(
        loc=torch.randn(8, 6),
        scale=torch.ones(8, 6),
        shape=(8,),
        device='cpu'
    )
    
    all_batches = torch.cat([batch_1, batch_2, batch_3], dim=0)
    print(f"Triple concatenation: {all_batches.shape}")
    assert all_batches.shape == (48,)               # 16 + 24 + 8 = 48
    print()


def demonstrate_multi_agent_scenarios():
    """Show multi-agent combination patterns."""
    print("=== Multi-Agent Scenarios ===")
    
    # Different agents with different action spaces
    agent_1_dist = TensorNormal(
        loc=torch.randn(10, 3),              # Agent 1: 10 steps, 3D actions
        scale=torch.ones(10, 3) * 0.4,
        shape=(10,),
        device='cpu'
    )
    
    agent_2_dist = TensorNormal(
        loc=torch.randn(10, 3),              # Agent 2: 10 steps, 3D actions
        scale=torch.ones(10, 3) * 0.6,      # Different exploration
        shape=(10,),
        device='cpu'
    )
    
    agent_3_dist = TensorNormal(
        loc=torch.randn(10, 3),              # Agent 3: 10 steps, 3D actions
        scale=torch.ones(10, 3) * 0.2,      # Conservative exploration
        shape=(10,),
        device='cpu'
    )
    
    # Stack agents for joint action sampling
    multi_agent_system = torch.stack([agent_1_dist, agent_2_dist, agent_3_dist], dim=0)
    print(f"Multi-agent system: {multi_agent_system.shape}")
    assert multi_agent_system.shape == (3, 10)      # 3 agents, 10 timesteps
    
    # Sample joint actions
    joint_actions = multi_agent_system.sample()     # Shape: (3, 10, 3)
    print(f"Joint actions: {joint_actions.shape}")
    assert joint_actions.shape == (3, 10, 3)
    
    # Different stacking dimension for timestep-wise analysis
    timestep_stacked = torch.stack([agent_1_dist, agent_2_dist], dim=1)
    print(f"Timestep-wise stacking: {timestep_stacked.shape}")
    assert timestep_stacked.shape == (10, 2)        # 10 timesteps, 2 agents
    
    # Concatenate agents across time (sequential execution)
    sequential_agents = torch.cat([agent_1_dist, agent_2_dist, agent_3_dist], dim=0)
    print(f"Sequential agents: {sequential_agents.shape}")
    assert sequential_agents.shape == (30,)          # 10 + 10 + 10 = 30 timesteps
    print()


def demonstrate_ensemble_methods():
    """Show ensemble and mixture combination patterns."""
    print("=== Ensemble Methods ===")
    
    # Create ensemble of different model outputs
    models = []
    for i in range(5):
        model_dist = TensorNormal(
            loc=torch.randn(20, 8) * (i + 1) * 0.1,  # Different model behaviors
            scale=torch.ones(20, 8) * (0.2 + i * 0.1),
            shape=(20,),
            device='cpu'
        )
        models.append(model_dist)
    
    # Stack ensemble for comparison
    ensemble = torch.stack(models, dim=0)
    print(f"Ensemble: {ensemble.shape}")
    assert ensemble.shape == (5, 20)                # 5 models, 20 states
    
    # Sample from all models
    ensemble_samples = ensemble.sample()            # Shape: (5, 20, 8)
    print(f"Ensemble samples: {ensemble_samples.shape}")
    
    # Create mixture weights
    mixture_weights = torch.softmax(torch.randn(5), dim=0)
    print(f"Mixture weights: {mixture_weights}")
    
    # Ensemble consensus (weighted average of means)
    ensemble_means = ensemble.mean                   # Shape: (5, 20, 8)
    consensus_mean = torch.sum(mixture_weights.view(5, 1, 1) * ensemble_means, dim=0)
    print(f"Consensus mean shape: {consensus_mean.shape}")
    
    # Create consensus distribution
    consensus_dist = TensorNormal(
        loc=consensus_mean,
        scale=torch.ones(20, 8) * 0.3,              # Reduced uncertainty
        shape=(20,),
        device='cpu'
    )
    
    print(f"Consensus distribution: {consensus_dist.shape}")
    print()


def demonstrate_hierarchical_composition():
    """Show hierarchical composition patterns."""
    print("=== Hierarchical Composition ===")
    
    # Level 1: Individual components
    component_a = TensorNormal(
        loc=torch.randn(5, 2),
        scale=torch.ones(5, 2),
        shape=(5,),
        device='cpu'
    )
    
    component_b = TensorNormal(
        loc=torch.randn(5, 2),
        scale=torch.ones(5, 2),
        shape=(5,),
        device='cpu'
    )
    
    # Level 2: Combine components into subsystems
    subsystem_1 = torch.stack([component_a, component_b], dim=1)
    print(f"Subsystem 1: {subsystem_1.shape}")
    assert subsystem_1.shape == (5, 2)              # 5 states, 2 components
    
    # Create another subsystem
    component_c = TensorNormal(
        loc=torch.randn(5, 2),
        scale=torch.ones(5, 2),
        shape=(5,),
        device='cpu'
    )
    
    component_d = TensorNormal(
        loc=torch.randn(5, 2),
        scale=torch.ones(5, 2),
        shape=(5,),
        device='cpu'
    )
    
    subsystem_2 = torch.stack([component_c, component_d], dim=1)
    
    # Level 3: Combine subsystems into full system
    full_system = torch.stack([subsystem_1, subsystem_2], dim=0)
    print(f"Full system: {full_system.shape}")
    assert full_system.shape == (2, 5, 2)           # 2 subsystems, 5 states, 2 components
    
    # Sample from hierarchical system
    system_samples = full_system.sample()           # Shape: (2, 5, 2, 2)
    print(f"Hierarchical samples: {system_samples.shape}")
    
    # Access different levels
    subsystem_1_reconstructed = full_system[0]      # Shape: (5, 2)
    component_a_reconstructed = full_system[0, :, 0] # Shape: (5,)
    
    print(f"Subsystem access: {subsystem_1_reconstructed.shape}")
    print(f"Component access: {component_a_reconstructed.shape}")
    print()


def demonstrate_mixed_distribution_types():
    """Show combinations of different distribution types.""" 
    print("=== Mixed Distribution Types ===")
    
    # Continuous action policy
    continuous_policy = TensorNormal(
        loc=torch.randn(16, 4),
        scale=torch.ones(16, 4) * 0.3,
        shape=(16,),
        device='cpu'
    )
    
    # Discrete action policy (for comparison)
    discrete_policy = TensorCategorical(
        logits=torch.randn(16, 6),           # 6 discrete actions
        shape=(16,),
        device='cpu'
    )
    
    # Can't directly stack different distribution types,
    # but can organize them in structures
    print(f"Continuous policy: {continuous_policy.shape}")
    print(f"Discrete policy: {discrete_policy.shape}")
    
    # Sample from both for comparison
    continuous_actions = continuous_policy.sample()  # Shape: (16, 4)
    discrete_actions = discrete_policy.sample()      # Shape: (16,)
    
    print(f"Continuous actions: {continuous_actions.shape}")
    print(f"Discrete actions: {discrete_actions.shape}")
    
    # Stack same types from different sources
    continuous_policy_2 = TensorNormal(
        loc=torch.randn(16, 4),
        scale=torch.ones(16, 4) * 0.5,
        shape=(16,),
        device='cpu'
    )
    
    continuous_comparison = torch.stack([continuous_policy, continuous_policy_2], dim=0)
    print(f"Continuous policy comparison: {continuous_comparison.shape}")
    print()


def main() -> None:
    """Demonstrate stacking and concatenation with TensorDistribution."""
    
    demonstrate_basic_stacking()
    demonstrate_concatenation()
    demonstrate_multi_agent_scenarios()
    demonstrate_ensemble_methods()
    demonstrate_hierarchical_composition()
    demonstrate_mixed_distribution_types()
    
    print("=== Key Insights ===")
    print("1. torch.stack creates new dimensions for comparison/ensemble")
    print("2. torch.cat extends existing dimensions for batch combination")
    print("3. Perfect for multi-agent, multi-task, and ensemble scenarios")
    print("4. Enables hierarchical composition of distribution systems")
    print("5. Natural integration with existing PyTorch tensor operations")
    print("6. Essential for complex ML architectures and model comparison")


if __name__ == "__main__":
    main()