import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_seed():
    torch.manual_seed(0)


def normalize_device(dev: torch.device) -> torch.device:
    d = torch.device(dev)
    # If no index was given, fill in current_device() for CUDA, leave CPU as-is
    if d.type == "cuda" and d.index is None:
        idx = torch.cuda.current_device()  # e.g. 0
        return torch.device(f"cuda:{idx}")
    return d
