"""
Basic TensorDistribution usage.

Key concept: TensorDistribution makes distributions behave like tensors,
allowing you to work with batches of distributions as single objects.
This example demonstrates the core insight that transforms probabilistic
modeling from manual parameter management to intuitive tensor operations.

Key concepts demonstrated:
- Creating distributions with batch dimensions
- Basic sampling operations (sample vs rsample)  
- Probability computation (log_prob)
- Shape validation and error handling
- The fundamental advantage over torch.distributions
"""

import torch
from tensorcontainer.tensor_distribution import TensorNormal


def main() -> None:
    """Demonstrate basic TensorDistribution functionality."""
    # Create a batch of 32 scalar normal distributions  
    # This represents 32 different states, each producing scalar values
    action_means = torch.randn(32)
    action_stds = torch.ones(32)
    
    # The key insight: one object represents 32 distributions
    # Shape and device are automatically inferred from the parameters
    action_dist = TensorNormal(
        loc=action_means,
        scale=action_stds
    )
    
    # Verify the distribution represents what we expect
    assert action_dist.shape == (32,)
    assert action_dist.batch_shape == (32,)  
    assert action_dist.event_shape == ()     # Each distribution outputs scalars
    
    # Sample actions for all 32 states at once
    actions = action_dist.sample()           # Shape: (32,)
    assert actions.shape == (32,)
    
    # Sample multiple action candidates for each state  
    action_candidates = action_dist.sample((10,))  # Shape: (10, 32)
    assert action_candidates.shape == (10, 32)
    
    # Use reparameterized sampling for gradient computation
    # This is crucial for training - gradients flow through the distribution
    action_means_grad = torch.randn(32, requires_grad=True)
    action_stds_grad = torch.ones(32)
    
    trainable_dist = TensorNormal(
        loc=action_means_grad,
        scale=action_stds_grad
    )
    
    # rsample enables gradient flow (essential for VAEs, policy gradients)
    reparameterized_actions = trainable_dist.rsample()
    assert reparameterized_actions.requires_grad  # Gradients preserved
    
    # Compute log probabilities for the sampled actions
    log_probs = action_dist.log_prob(actions)    # Shape: (32,)
    assert log_probs.shape == (32,)
    
    # Access distribution properties
    mean_actions = action_dist.mean             # Shape: (32,)
    action_variance = action_dist.variance      # Shape: (32,)
    action_entropy = action_dist.entropy()     # Shape: (32,)
    
    assert mean_actions.shape == (32,)
    assert action_variance.shape == (32,)
    assert action_entropy.shape == (32,)
    
    # Demonstrate shape validation - this is where TensorDistribution shines
    # Parameter shapes must be broadcastable
    try:
        # This will fail because parameters have incompatible shapes
        TensorNormal(
            loc=torch.randn(16),    # Shape: (16,) 
            scale=torch.ones(20)    # Shape: (20,) - not broadcastable!
        )
    except Exception as e:
        # Expected error due to shape mismatch
        print(f"Shape validation caught error: {type(e).__name__}")
    
    # Show the power: this is much simpler than torch.distributions
    # With torch.distributions, you'd need to manually track and manage
    # all parameters for device movement, reshaping, etc.
    print(f"Created {action_dist.shape[0]} distributions as a single object")
    print(f"Sampled actions shape: {actions.shape}")
    print(f"Mean action values: {mean_actions}")  # All distribution means


if __name__ == "__main__":
    main()