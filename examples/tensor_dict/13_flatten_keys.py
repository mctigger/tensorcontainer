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

    # Custom separator
    underscore = nested.flatten_keys(separator="_")
    assert "env_obs" in underscore
    assert "env_info_step" in underscore

    # Operations work on flattened structure
    reshaped_flat = flattened.reshape(1, 3)
    assert reshaped_flat["env.obs"].shape == (1, 3, 10)


if __name__ == "__main__":
    main()