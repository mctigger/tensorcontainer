# TensorContainer Examples

This directory contains comprehensive examples demonstrating the capabilities of TensorContainer's main components.

## Overview

- **[tensor_dataclass/](tensor_dataclass/)** - Examples for `TensorDataClass` with static, type-safe schemas
- **[tensor_dict/](tensor_dict/)** - Examples for `TensorDict` with dynamic dictionary-style access

## Getting Started

Each example is self-contained and can be run directly:

```bash
# Run individual examples
python tensor_dataclass/01_basic.py
python tensor_dict/01_basic.py
```

## Example Categories

### TensorDataClass Examples (tensor_dataclass/)
Static schemas with compile-time type checking:
- `01_basic.py` - Basic usage and validation
- `02_indexing.py` - Indexing and slicing operations
- `03_shape_ops.py` - Shape transformations (reshape, squeeze, etc.)
- `04_stack.py` - Stacking and concatenation
- `05_device.py` - Moving between CPU/GPU
- `06_inheritance.py` - Class inheritance patterns
- `07_copy_clone.py` - Shallow vs deep copying
- `08_detach_gradients.py` - Gradient management
- `09_nested.py` - Nested container structures

### TensorDict Examples (tensor_dict/)
Dynamic schemas with runtime flexibility:
- `01_basic.py` - Basic dictionary-style usage
- `02_indexing.py` - Indexing and slicing
- `03_shape_ops.py` - Shape operations
- `04_stack.py` - Combining multiple TensorDicts
- `05_device.py` - Device management
- `07_copy_clone.py` - Copy operations
- `08_detach_gradients.py` - Gradient handling
- `09_nested.py` - Nested structures
- `10_dynamic_keys.py` - Runtime key management
- `11_mapping_interface.py` - Dictionary-like methods
- `12_nested_dict_handling.py` - Automatic dict conversion
- `13_flatten_keys.py` - Key flattening utilities

