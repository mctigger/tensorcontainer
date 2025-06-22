import functools

import pytest
import torch
from torch._inductor import exc as inductor_exc


@pytest.fixture
def nested_dict():
    def _make(shape):
        nested_dict_data = {
            "x": {
                "a": torch.arange(0, 4).reshape(*shape),
                "b": torch.arange(4, 8).reshape(*shape),
            },
            "y": torch.arange(8, 12).reshape(*shape),
        }
        return nested_dict_data

    return _make


@functools.lru_cache(None)
def has_cpp_compiler():
    """
    Checks if a C++ compiler is available for torch.compile.
    Caches the result to avoid re-running this check for every test.
    """
    try:
        # A minimal function to test compilation
        def dummy_fn(x):
            return x + 1

        # Attempt to compile and run the function
        compiled_fn = torch.compile(dummy_fn)
        compiled_fn(torch.randn(1))
        return True
    except (inductor_exc.InductorError, RuntimeError) as e:
        # Check if the error message indicates a missing compiler
        if "InvalidCxxCompiler" in str(e) or ("C++" in str(e) and "compiler" in str(e)):
            return False
        # Re-raise the exception if it's not the one we're looking for
        raise


skipif_no_compile = pytest.mark.skipif(
    not has_cpp_compiler(),
    reason="Test requires a C++ compiler for torch.compile, which was not found.",
)


def pytest_configure(config):
    """
    Pytest hook to dynamically register markers.
    """
    config.addinivalue_line(
        "markers",
        "skipif_no_compile: skip test if C++ compiler is not available",
    )
