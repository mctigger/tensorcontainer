"""
TensorDict key flattening.

Key concept: flatten_keys() converts nested hierarchical keys into
flat dot-separated keys while sharing tensor memory.
"""

import torch
from tensorcontainer import TensorDict


def main() -> None:
    nested = TensorDict(
        {
            "env": {
                "obs": torch.randn(3, 10),
                "info": {"step": torch.tensor([1, 2, 3])},
            },
            "agent": {"policy": torch.randn(3, 6)},
        },
        shape=(3,),
        device="cpu",
    )

    # Flatten with default dot separator
    flattened = nested.flatten_keys()
    assert "env.obs" in flattened
    assert "env.info.step" in flattened
    assert "agent.policy" in flattened
    
    # Memory is shared between original and flattened
    assert nested["env"]["obs"] is flattened["env.obs"]
    assert torch.equal(nested["env"]["info"]["step"], flattened["env.info.step"])


if __name__ == "__main__":
    main()