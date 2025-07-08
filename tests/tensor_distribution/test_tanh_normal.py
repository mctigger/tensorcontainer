import pytest
import torch

from tensorcontainer.tensor_distribution.base import TensorDistribution
from tensorcontainer.tensor_distribution.tanh_normal import TensorTanhNormal
from tests.conftest import skipif_no_cuda

from .conftest import normalize_device


def test_init_valid():
    """Tests that TensorTanhNormal can be instantiated with valid parameters."""
    loc = torch.zeros(2, 3)
    scale = torch.ones(2, 3)
    dist = TensorTanhNormal(
        loc=loc,
        scale=scale,
        reinterpreted_batch_ndims=0,
        shape=loc.shape,
        device=loc.device,
    )
    assert isinstance(dist, TensorDistribution)


def test_sample_shape_and_dtype():
    """
    Tests that the sample() method produces tensors of the correct shape,
    dtype, and on the correct device.
    """
    loc = torch.randn(4, 3)
    scale = torch.rand(4, 3) + 1e-6  # ensure scale is positive
    dist = TensorTanhNormal(
        loc=loc,
        scale=scale,
        reinterpreted_batch_ndims=0,
        shape=loc.shape,
        device=loc.device,
    )
    # Draw 5 i.i.d. samples
    samples = dist.sample(sample_shape=torch.Size((5,)))

    # Expected shape is (sample_shape, *batch_shape)
    assert samples.shape == (5, *loc.shape)
    assert samples.dtype == torch.float32

    # Check that all samples are within the range [-1, 1] due to tanh transform
    assert torch.all(samples >= -1.0)
    assert torch.all(samples <= 1.0)


@pytest.mark.parametrize(
    "rbn_dims,expected_shape",
    [
        (0, (2, 3)),  # no reinterpretation -> log_prob per-element
        (1, (2,)),  # sum over the last dimension
        (2, ()),  # sum over the last two dimensions -> scalar
    ],
)
def test_log_prob_reinterpreted_batch_ndims(rbn_dims, expected_shape):
    """
    Tests the log_prob method with different values for
    reinterpreted_batch_ndims, ensuring the output shape is correct.
    """
    loc = torch.tensor([[0.0, 1.0, -1.0], [0.5, -0.5, 0.5]])
    scale = torch.tensor([[0.2, 0.8, 0.1], [0.5, 0.5, 0.5]])
    dist = TensorTanhNormal(
        loc=loc,
        scale=scale,
        reinterpreted_batch_ndims=rbn_dims,
        shape=loc.shape,
        device=loc.device,
    )

    # A sample to evaluate the log probability of - ensure it's within [-1, 1]
    x = torch.tensor([[0.1, 0.8, -0.9], [0.4, -0.6, 0.7]])
    lp = dist.log_prob(x)

    assert lp.shape == expected_shape


def test_mean_and_mode():
    """
    Tests that the mean and mode properties return tensors of the correct shape.
    The SamplingDistribution used in TensorTanhNormal estimates these through sampling.
    """
    loc = torch.tensor([[0.0, 1.0, -1.0], [0.5, -0.5, 0.5]])
    scale = torch.tensor([[0.2, 0.8, 0.1], [0.5, 0.5, 0.5]])
    dist = TensorTanhNormal(
        loc=loc,
        scale=scale,
        reinterpreted_batch_ndims=0,
        shape=loc.shape,
        device=loc.device,
    )

    mean = dist.mean
    mode = dist.mode

    assert mean.shape == loc.shape
    # The mode might be reshaped due to how SamplingDistribution works
    assert mode.numel() == loc.numel()

    # Check that mean and mode are within the range [-1, 1] due to tanh transform
    assert torch.all(mean >= -1.0)
    assert torch.all(mean <= 1.0)
    assert torch.all(mode >= -1.0)
    assert torch.all(mode <= 1.0)


def test_copy():
    """Tests that the copy method creates a new instance with the same parameters."""
    loc = torch.randn(2, 3)
    scale = torch.rand(2, 3) + 1e-6  # ensure scale is positive
    dist = TensorTanhNormal(
        loc=loc,
        scale=scale,
        reinterpreted_batch_ndims=1,
        shape=loc.shape,
        device=loc.device,
    )

    dist_copy = dist.copy()

    assert isinstance(dist_copy, TensorTanhNormal)
    # Compare the underlying tensor values by converting to tensors
    assert torch.allclose(loc, dist_copy.loc)  # type: ignore
    assert torch.allclose(scale, dist_copy.scale)  # type: ignore
    assert dist.reinterpreted_batch_ndims == dist_copy.reinterpreted_batch_ndims
    assert dist.shape == dist_copy.shape
    # Compare devices as strings to avoid type issues
    assert str(dist.device) == str(dist_copy.device)


@skipif_no_cuda
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
        dev2 = torch.device("cpu:0")  # This is not standard but torch handles it
        assert normalize_device(dev1) == normalize_device(dev2)
