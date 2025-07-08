# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an academic research project implementing tensor container functionality for PyTorch. The main components are:

- **TensorContainer**: Base class for tensor containers with PyTree compatibility
- **TensorDict**: Dictionary-like container for nested tensors with PyTree support
- **TensorDataClass**: DataClass-based tensor container using @dataclass_transform
- **TensorDistribution**: Distribution wrapper for tensor containers

## Key Architecture

The project follows a layered architecture:

1. **Base Layer**: `TensorContainer` (src/rtd/tensor_container.py) - Provides core tensor manipulation operations
2. **Container Layer**: `TensorDict` and `TensorDataClass` - Specific implementations for different use cases
3. **Distribution Layer**: `TensorDistribution` - Wraps distributions for tensor containers
4. **PyTree Integration**: All classes implement PyTree compatibility via `torch.utils._pytree`

### Core Concepts

- **PyTree Compatibility**: All tensor containers are registered as PyTrees for seamless integration with torch.compile
- **Shape/Device Consistency**: All containers maintain consistent shape and device attributes across nested tensors
- **Torch Function Override**: Custom implementations of torch functions via `__torch_function__` protocol
- **Event Dimensions**: Support for event dimensions in tensor operations

## Development Commands

### Testing
```bash
# Run all tests with coverage
pytest --strict-markers --cov=src

# Run specific test module
pytest tests/tensor_dict/test_compile.py

# Run single test
pytest tests/tensor_dict/test_compile.py::TestCompile::test_basic_compile
```

### Development Setup
```bash
# Install development dependencies
pip install -e .[dev]
```

## Testing Guidelines

This project heavily emphasizes `torch.compile` compatibility. Key testing patterns:

1. **Central Verification Helper Pattern**: Each test class has a helper method (e.g., `_run_and_verify_setitem`) that:
   - Tests both eager and compiled execution
   - Verifies torch.compile compatibility with graph break counting
   - Compares results between eager and compiled modes
   - Validates shape, device, and value correctness

2. **Compile Testing Utilities**: 
   - `run_and_compare_compiled()` - Runs operations in both modes and compares results
   - `assert_tc_equal()` - Compares TensorContainer instances comprehensively
   - Graph break counting for torch.compile compatibility

3. **Test Structure**:
   - Use parametrized tests with descriptive IDs
   - Group test cases by behavior (basic indexing, advanced indexing, etc.)
   - Include docstrings with torch.Tensor examples
   - Test error conditions with `pytest.raises(ErrorType, match="...")`

## Important Files

- `src/rtd/tensor_container.py` - Base container class with core operations
- `src/rtd/tensor_dict.py` - Dictionary-style tensor container
- `src/rtd/tensor_dataclass.py` - DataClass-style tensor container
- `tests/compile_utils.py` - Utilities for testing torch.compile compatibility
- `docs/testing.md` - Comprehensive testing guidelines and patterns

## Configuration

Global configuration in `src/rtd/config.py`:
- `validate_args: bool = True` - Controls argument validation

## Key Dependencies

- PyTorch (torch) - Core tensor operations
- torch.utils._pytree - PyTree functionality for nested structures
- typing_extensions - For @dataclass_transform decorator