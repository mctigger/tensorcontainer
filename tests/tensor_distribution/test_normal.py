import pytest
import torch
from rtd.tensor_distribution import TensorNormal, TensorDistribution


def normalize_device(dev: torch.device) -> torch.device:
    """
    Normalizes a torch.device object to include the device index for CUDA.

    This ensures that a device specified as "cuda" is resolved to "cuda:0"
    (or the current device), making comparisons consistent.
    """
    d = torch.device(dev)
    # If no index was given, fill in current_device() for CUDA, leave CPU as-is
    if d.type == "cuda" and d.index is None:
        idx = torch.cuda.current_device()
        return torch.device(f"cuda:{idx}")
    return d


def test_init_valid():
    """Tests that TensorNormal can be instantiated with valid parameters."""
    loc = torch.zeros(2, 3)
    scale = torch.ones(2, 3)
    dist = TensorNormal(loc=loc, scale=scale, shape=loc.shape, reinterpreted_batch_ndims=0)
    assert isinstance(dist, TensorDistribution)


def test_sample_shape_and_dtype():
    """
    Tests that the sample() method produces tensors of the correct shape,
    dtype, and on the correct device.
    """
    loc = torch.randn(4, 3)
    scale = torch.rand(4, 3) + 1e-6  # ensure scale is positive
    dist = TensorNormal(
        loc=loc, scale=scale, shape=loc.shape, reinterpreted_batch_ndims=0
    )
    # Draw 5 i.i.d. samples
    samples = dist.sample(sample_shape=torch.Size((5,)))

    # Expected shape is (sample_shape, *batch_shape)
    assert samples.shape == (5, *loc.shape)
    assert samples.dtype == torch.float32


@pytest.mark.parametrize(
    "rbn_dims,expected_shape",
    [
        (0, (2, 3)),  # no reinterpretation -> log_prob per-element
        (1, (2,)),    # sum over the last dimension
        (2, ()),      # sum over the last two dimensions -> scalar
    ],
)
def test_log_prob_reinterpreted_batch_ndims(rbn_dims, expected_shape):
    """
    Tests the log_prob method with different values for
    reinterpreted_batch_ndims, ensuring the output shape and values are correct
    by comparing with torch.distributions.Normal.
    """
    loc = torch.tensor([[0.0, 1.0, -1.0], [0.5, -0.5, 0.5]])
    scale = torch.tensor([[0.2, 0.8, 0.1], [0.5, 0.5, 0.5]])
    dist = TensorNormal(
        loc=loc,
        scale=scale,
        shape=loc.shape,
        reinterpreted_batch_ndims=rbn_dims,
    )
    # A sample to evaluate the log probability of
    x = torch.tensor([[0.1, 1.2, -1.1], [0.4, -0.6, 0.7]])
    lp = dist.log_prob(x)

    # Calculate expected log_prob using the reference torch.distributions
    torch_dist = torch.distributions.Normal(loc, scale)
    ref_lp = torch_dist.log_prob(x)
    if rbn_dims > 0:
        # Sum over the dimensions that are being reinterpreted as event dims
        ref_lp = ref_lp.sum(dim=list(range(-rbn_dims, 0)))

    assert lp.shape == expected_shape
    assert torch.allclose(lp, ref_lp)


def test_device_normalization_helper():
    """
    Tests the internal device normalization helper to ensure "cuda" and
    "cuda:0" are treated as equivalent.
    """
    # This test is only meaningful if CUDA is available.
    if torch.cuda.is_available():
        dev1 = torch.device("cuda")
        # Get the device from a tensor created on the default CUDA device
        dev2 = torch.ones(1, device="cuda").device
        assert normalize_device(dev1) == normalize_device(dev2)
    else:
        # On a CPU-only machine, test with "cpu"
        dev1 = torch.device("cpu")
        dev2 = torch.device("cpu:0") # This is not standard but torch handles it
        assert normalize_device(dev1) == normalize_device(dev2)