"""
Example demonstrating the complexity of method chaining with torch.distributions.Normal.

This example shows how standard torch.distributions.Normal requires manual parameter
manipulation to achieve the same transformations that TensorNormal handles seamlessly.
It highlights the verbose, error-prone nature of working with standard distributions
when you need to apply tensor operations to distribution parameters.
"""
import torch
from torch.distributions import Normal


def demonstrate_torch_normal_chaining():
    """
    Demonstrate the manual work required for chaining operations on torch.distributions.Normal.
    
    Shows how each transformation requires explicit parameter extraction, transformation,
    and reconstruction of the distribution object.
    """
    print("=== torch.distributions.Normal Method Chaining Demo ===\n")
    
    # Create a Normal distribution with batch shape (2, 3) and gradients enabled
    print("1. Creating Normal distribution with batch shape (2, 3)")
    loc = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], requires_grad=True)
    scale = torch.tensor([[1.0, 0.5, 2.0], [1.5, 0.8, 1.2]], requires_grad=True)
    
    normal = Normal(loc=loc, scale=scale)
    print(f"Original batch shape: {normal.batch_shape}")
    print(f"Original loc shape: {normal.loc.shape}")
    print(f"Original scale shape: {normal.scale.shape}")
    print(f"Requires grad - loc: {normal.loc.requires_grad}, scale: {normal.scale.requires_grad}")
    print()
    
    # Manual chaining: view -> detach -> permute -> expand
    print("2. Manual chaining operations: view(6) -> detach() -> permute(0) -> expand(2, 6)")
    print("   Each step requires manual parameter extraction and reconstruction!")
    
    print("\n   Step 1: Manual .view(6) - reshape batch dimensions")
    # Extract parameters, transform them, create new distribution
    viewed_loc = normal.loc.view(6)
    viewed_scale = normal.scale.view(6)
    viewed_normal = Normal(loc=viewed_loc, scale=viewed_scale)
    print(f"   After manual view: batch_shape={viewed_normal.batch_shape}")
    print(f"   Required code: Normal(loc=normal.loc.view(6), scale=normal.scale.view(6))")
    
    print("\n   Step 2: Manual .detach() - remove gradients")
    # Extract parameters, detach them, create new distribution
    detached_loc = viewed_normal.loc.detach()
    detached_scale = viewed_normal.scale.detach()
    detached_normal = Normal(loc=detached_loc, scale=detached_scale)
    print(f"   After manual detach: batch_shape={detached_normal.batch_shape}")
    print(f"   Requires grad - loc: {detached_normal.loc.requires_grad}, scale: {detached_normal.scale.requires_grad}")
    print(f"   Required code: Normal(loc=viewed_normal.loc.detach(), scale=viewed_normal.scale.detach())")
    
    print("\n   Step 3: Manual .permute(0) - permute batch dimensions")
    # Extract parameters, permute them, create new distribution
    permuted_loc = detached_normal.loc.permute(0)  # Identity for 1D
    permuted_scale = detached_normal.scale.permute(0)
    permuted_normal = Normal(loc=permuted_loc, scale=permuted_scale)
    print(f"   After manual permute: batch_shape={permuted_normal.batch_shape}")
    print(f"   Required code: Normal(loc=detached_normal.loc.permute(0), scale=detached_normal.scale.permute(0))")
    
    print("\n   Step 4: Manual .expand(2, 6) - expand batch dimensions")
    # Extract parameters, expand them, create new distribution
    expanded_loc = permuted_normal.loc.expand(2, 6)
    expanded_scale = permuted_normal.scale.expand(2, 6)
    expanded_normal = Normal(loc=expanded_loc, scale=expanded_scale)
    print(f"   After manual expand: batch_shape={expanded_normal.batch_shape}")
    print(f"   Required code: Normal(loc=permuted_normal.loc.expand(2, 6), scale=permuted_normal.scale.expand(2, 6))")
    print()
    
    # Show the verbose way to do it all at once
    print("3. All operations done manually in one expression:")
    print("   (Very verbose and error-prone!)")
    
    manual_chained = Normal(
        loc=normal.loc.view(6).detach().permute(0).expand(2, 6),
        scale=normal.scale.view(6).detach().permute(0).expand(2, 6)
    )
    print(f"Manual chained result batch_shape: {manual_chained.batch_shape}")
    print(f"Manual chained loc shape: {manual_chained.loc.shape}")
    print(f"Manual chained scale shape: {manual_chained.scale.shape}")
    print(f"Requires grad - loc: {manual_chained.loc.requires_grad}, scale: {manual_chained.scale.requires_grad}")
    print()
    
    # Demonstrate that the distribution interface still works
    print("4. Distribution interface still works after manual chaining:")
    samples = manual_chained.sample()
    log_probs = manual_chained.log_prob(samples)
    entropy = manual_chained.entropy()
    
    print(f"Sample shape: {samples.shape}")
    print(f"Log prob shape: {log_probs.shape}")
    print(f"Entropy shape: {entropy.shape}")
    print(f"Mean shape: {manual_chained.mean.shape}")
    print(f"Variance shape: {manual_chained.variance.shape}")
    print()
    
    # Show the complexity with 3D batch shapes
    print("5. Complex manual chaining with 3D batch shape:")
    print("   (Even more verbose and error-prone!)")
    
    # Create 3D batch shape (2, 2, 3)
    loc_3d = torch.randn(2, 2, 3, requires_grad=True)
    scale_3d = torch.abs(torch.randn(2, 2, 3, requires_grad=True)) + 0.1
    
    normal_3d = Normal(loc=loc_3d, scale=scale_3d)
    print(f"Original 3D batch_shape: {normal_3d.batch_shape}")
    
    # Manual chain: reshape -> permute -> detach -> expand
    # Each parameter must be transformed separately!
    complex_manual_chained = Normal(
        loc=(normal_3d.loc
             .view(4, 3)           # Reshape first two dims
             .permute(1, 0)        # Swap dimensions  
             .detach()             # Remove gradients
             .expand(3, 4, 2)),    # Expand with new dimension
        scale=(normal_3d.scale
               .view(4, 3)         # Reshape first two dims
               .permute(1, 0)      # Swap dimensions  
               .detach()           # Remove gradients
               .expand(3, 4, 2))   # Expand with new dimension
    )
    
    print(f"Complex manual chained batch_shape: {complex_manual_chained.batch_shape}")
    print(f"Complex manual chained loc shape: {complex_manual_chained.loc.shape}")
    print(f"Still works as distribution: sample shape = {complex_manual_chained.sample().shape}")
    print()
    
    print("6. Problems with the manual approach:")
    print("   - Verbose: Every parameter must be transformed separately")
    print("   - Error-prone: Easy to forget a parameter or apply wrong transformation")
    print("   - Not scalable: More parameters = more manual work")
    print("   - Type-specific: Different distributions need different parameter handling")
    print("   - No method chaining: Can't use fluent interface patterns")
    print()
    
    print("7. What happens if you forget to transform a parameter?")
    print("   (This is a common source of bugs!)")
    
    try:
        # Oops! Forgot to transform the scale parameter
        buggy_distribution = Normal(
            loc=normal.loc.view(6).detach(),
            scale=normal.scale  # BUG: Forgot to apply transformations!
        )
        # This will fail when we try to use it
        buggy_sample = buggy_distribution.sample()
        print("   This should have failed but didn't - that's concerning!")
    except RuntimeError as e:
        print(f"   RuntimeError (as expected): {e}")
    except Exception as e:
        print(f"   Unexpected error: {type(e).__name__}: {e}")
    
    print("\n8. Comparison summary:")
    print("   TensorNormal:        normal.view(6).detach().permute(0).expand(2, 6)")
    print("   torch.Normal:        Normal(")
    print("                            loc=normal.loc.view(6).detach().permute(0).expand(2, 6),")
    print("                            scale=normal.scale.view(6).detach().permute(0).expand(2, 6)")
    print("                        )")
    print("   TensorNormal is clearly more concise and less error-prone!")


