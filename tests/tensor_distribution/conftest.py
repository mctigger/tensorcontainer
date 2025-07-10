import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_seed():
    torch.manual_seed(0)


@pytest.fixture(autouse=True)
def preserve_distributions_validation():
    """
    Preserve and restore torch.distributions validation state around each test.
    This prevents torch.compile usage in one test from affecting subsequent tests.

    This fixture automatically runs before and after every test in this directory,
    ensuring that any global state changes (like torch.compile disabling validation)
    don't leak between tests.
    """
    # Store the original validation state before the test
    original_validate_args = torch.distributions.Distribution._validate_args

    yield  # Run the test

    # Restore the original validation state after the test
    torch.distributions.Distribution.set_default_validate_args(original_validate_args)


@pytest.fixture(autouse=True)
def with_distributions_validation():
    """
    Fixture to ensure distributions validation is enabled for specific tests.

    Usage:
        def test_validation_required(self, with_distributions_validation):
            # Validation is guaranteed to be enabled in this test
            with pytest.raises(ValueError):
                TensorNormal(loc=torch.tensor([1.0]), scale=torch.tensor([-0.1]))
    """
    # Force enable validation for this test
    torch.distributions.Distribution.set_default_validate_args(True)
    yield


def normalize_device(dev: torch.device) -> torch.device:
    d = torch.device(dev)
    # If no index was given, fill in current_device() for CUDA, leave CPU as-is
    if d.type == "cuda" and d.index is None:
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()  # e.g. 0
            return torch.device(f"cuda:{idx}")
        else:
            # If CUDA is not available, return the device as-is
            return d
    return d
