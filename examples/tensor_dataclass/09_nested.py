"""
Nesting TensorDataClass instances.

This example demonstrates how to nest TensorDataClass instances within each
other to create complex, hierarchical data structures. Operations on the
outer instance, such as moving to a different device, will propagate
recursively to all nested instances.

Key concepts demonstrated:
- Nested structure: How to define a TensorDataClass that contains another
  TensorDataClass as a field.
- Recursive operations: How operations on the parent instance are
  automatically applied to all nested instances, ensuring consistency.
"""

import torch
from tensorcontainer import TensorDataClass


class AgentState(TensorDataClass):
    actor_params: torch.Tensor
    critic_params: torch.Tensor


class FullState(TensorDataClass):
    env_state: torch.Tensor
    agent_state: AgentState  # Nested TensorDataClass


def main() -> None:
    """Demonstrate nesting TensorDataClass instances."""
    # Create with appropriate tensors and shapes
    full = FullState(
        env_state=torch.randn(4, 10),
        agent_state=AgentState(
            actor_params=torch.randn(4, 20),
            critic_params=torch.randn(4, 15),
            shape=(4,),
            device="cpu",
        ),
        shape=(4,),
        device="cpu",
    )

    # Operations on 'full' will propagate to 'agent_state'
    # For example, moving to a different device
    full_cuda = full.to("cuda")
    assert full_cuda.device == torch.device("cuda:0")
    assert full_cuda.agent_state.device == torch.device("cuda:0")
    assert full_cuda.agent_state.actor_params.device == torch.device("cuda:0")


if __name__ == "__main__":
    main()
