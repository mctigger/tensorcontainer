"""Tests for unsafe construction context manager in TensorContainer."""

import pytest
import torch

from tensorcontainer.tensor_container import TensorContainer
from tensorcontainer.tensor_dataclass import TensorDataClass
from tensorcontainer.tensor_dict import TensorDict
from tests.conftest import skipif_no_cuda


@pytest.fixture
def sample_tensors():
    """Create sample tensors for testing."""
    return {
        "features": torch.randn(4, 10),
        "labels": torch.randn(4, 1),
        "incompatible_features": torch.randn(3, 10),  # Wrong batch size
        "incompatible_labels": torch.randn(5, 1),  # Wrong batch size
    }


class SampleTensorDataClass(TensorDataClass):
    """Test TensorDataClass for unsafe construction."""

    features: torch.Tensor
    labels: torch.Tensor


def test_normal_construction_with_validation(sample_tensors):
    """Test that normal construction validates tensors."""
    # This should work - compatible tensors
    container = SampleTensorDataClass(
        features=sample_tensors["features"],
        labels=sample_tensors["labels"],
        shape=(4,),
        device=torch.device("cpu"),
    )
    assert container.shape == (4,)
    assert container.device == torch.device("cpu")

    # This should fail - incompatible shape
    with pytest.raises(RuntimeError, match="Invalid shape"):
        SampleTensorDataClass(
            features=sample_tensors["incompatible_features"],  # Wrong batch size
            labels=sample_tensors["labels"],
            shape=(4,),
            device=torch.device("cpu"),
        )


def test_unsafe_construction_skips_validation(sample_tensors):
    """Test that unsafe construction skips validation."""
    # This should fail with normal construction
    with pytest.raises(RuntimeError, match="Invalid shape"):
        SampleTensorDataClass(
            features=sample_tensors["incompatible_features"],
            labels=sample_tensors["incompatible_labels"],
            shape=(4,),
            device=torch.device("cpu"),
        )

    # But should succeed with unsafe construction
    with TensorContainer.unsafe_construction():
        container = SampleTensorDataClass(
            features=sample_tensors["incompatible_features"],
            labels=sample_tensors["incompatible_labels"],
            shape=(4,),
            device=torch.device("cpu"),
        )
        assert container.shape == (4,)  # Shape is set as requested
        assert container.features.shape == (
            3,
            10,
        )  # But actual data has different shape
        assert container.labels.shape == (5, 1)


@skipif_no_cuda
def test_unsafe_construction_with_device_mismatch(sample_tensors):
    """Test unsafe construction with device mismatch."""
    # This should fail with normal construction due to device mismatch
    with pytest.raises(RuntimeError, match="Invalid device"):
        SampleTensorDataClass(
            features=sample_tensors["features"],
            labels=sample_tensors["labels"],
            shape=(4,),
            device=torch.device("cuda"),
        )

    # But should succeed with unsafe construction
    with TensorContainer.unsafe_construction():
        container = SampleTensorDataClass(
            features=sample_tensors["features"],
            labels=sample_tensors["labels"],
            shape=(4,),
            device=torch.device("cuda"),
        )
        assert (
            container.device is not None and container.device.type == "cuda"
        )  # Device is set as requested
        assert container.features.device == torch.device(
            "cpu"
        )  # But actual data is on CPU
        assert container.labels.device == torch.device("cpu")


def test_unsafe_construction_context_manager_isolation(sample_tensors):
    """Test that unsafe construction is properly isolated to the context."""
    # Before context: validation should work normally
    with pytest.raises(RuntimeError, match="Invalid shape"):
        SampleTensorDataClass(
            features=sample_tensors["incompatible_features"],
            labels=sample_tensors["labels"],
            shape=(4,),
            device=torch.device("cpu"),
        )

    # Inside context: validation should be disabled
    with TensorContainer.unsafe_construction():
        container = SampleTensorDataClass(
            features=sample_tensors["incompatible_features"],
            labels=sample_tensors["labels"],
            shape=(4,),
            device=torch.device("cpu"),
        )
        assert container is not None

    # After context: validation should work normally again
    with pytest.raises(RuntimeError, match="Invalid shape"):
        SampleTensorDataClass(
            features=sample_tensors["incompatible_features"],
            labels=sample_tensors["labels"],
            shape=(4,),
            device=torch.device("cpu"),
        )


def test_unsafe_construction_nested_contexts(sample_tensors):
    """Test that nested unsafe construction contexts work correctly."""
    with TensorContainer.unsafe_construction():
        # Should work in outer context
        SampleTensorDataClass(
            features=sample_tensors["incompatible_features"],
            labels=sample_tensors["labels"],
            shape=(4,),
            device=torch.device("cpu"),
        )

        with TensorContainer.unsafe_construction():
            # Should work in nested context
            container2 = SampleTensorDataClass(
                features=sample_tensors["incompatible_features"],
                labels=sample_tensors["incompatible_labels"],
                shape=(5,),
                device=torch.device("cpu"),
            )
            assert container2.shape == (5,)

        # Should still work in outer context after nested context exits
        container3 = SampleTensorDataClass(
            features=sample_tensors["incompatible_features"],
            labels=sample_tensors["labels"],
            shape=(6,),
            device=torch.device("cpu"),
        )
        assert container3.shape == (6,)

    # Should fail after all contexts exit
    with pytest.raises(RuntimeError, match="Invalid shape"):
        SampleTensorDataClass(
            features=sample_tensors["incompatible_features"],
            labels=sample_tensors["labels"],
            shape=(4,),
            device=torch.device("cpu"),
        )


def test_unsafe_construction_with_tensor_dict(sample_tensors):
    """Test unsafe construction with TensorDict."""
    # Create incompatible nested structure
    incompatible_data = {
        "features": sample_tensors["incompatible_features"],  # Shape (3, 10)
        "labels": sample_tensors["incompatible_labels"],  # Shape (5, 1)
    }

    # This should fail with normal construction
    with pytest.raises(RuntimeError, match="Invalid shape"):
        TensorDict(incompatible_data, shape=(4,), device=torch.device("cpu"))

    # But should succeed with unsafe construction
    with TensorContainer.unsafe_construction():
        container = TensorDict(
            incompatible_data, shape=(4,), device=torch.device("cpu")
        )
        assert container.shape == (4,)
        assert container["features"].shape == (3, 10)
        assert container["labels"].shape == (5, 1)


def test_unsafe_construction_thread_safety(sample_tensors):
    """Test that unsafe construction is thread-safe using thread-local storage."""
    import threading
    import time

    results = {}
    errors = {}

    def test_thread(thread_id, use_unsafe):
        try:
            if use_unsafe:
                with TensorContainer.unsafe_construction():
                    # Add small delay to test concurrency
                    time.sleep(0.01)
                    SampleTensorDataClass(
                        features=sample_tensors["incompatible_features"],
                        labels=sample_tensors["incompatible_labels"],
                        shape=(4,),
                        device=torch.device("cpu"),
                    )
                    results[thread_id] = "success"
            else:
                try:
                    SampleTensorDataClass(
                        features=sample_tensors["incompatible_features"],
                        labels=sample_tensors["incompatible_labels"],
                        shape=(4,),
                        device=torch.device("cpu"),
                    )
                    results[thread_id] = "unexpected_success"
                except RuntimeError:
                    results[thread_id] = "expected_failure"
        except Exception as e:
            errors[thread_id] = str(e)

    # Start multiple threads - some with unsafe construction, some without
    threads = []
    for i in range(10):
        use_unsafe = i % 2 == 0  # Every other thread uses unsafe construction
        thread = threading.Thread(target=test_thread, args=(i, use_unsafe))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Check results
    assert len(errors) == 0, f"Unexpected errors: {errors}"

    for i in range(10):
        if i % 2 == 0:  # Threads that used unsafe construction
            assert results[i] == "success", (
                f"Thread {i} should have succeeded with unsafe construction"
            )
        else:  # Threads that used normal construction
            assert results[i] == "expected_failure", (
                f"Thread {i} should have failed with normal construction"
            )


if __name__ == "__main__":
    pytest.main([__file__])
