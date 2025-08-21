# TensorDict Examples

This folder contains small, runnable scripts that demonstrate how to use TensorDict for common workflows. Each script can be executed directly:

- Basic usage: `python examples/tensor_dict/01_basic.py`

## Core TensorDict Examples
- 01_basic.py — Create a TensorDict, access fields with dictionary notation
- 02_indexing.py — Index and slice batch dimensions, dynamic key management
- 03_shape_ops.py — Reshape batch dimensions while preserving event dimensions
- 04_stack.py — Combine instances with torch.stack and torch.cat
- 05_device.py — Move instances between CPU and CUDA, handle mixed devices
- 06_inheritance.py — Custom TensorDict subclasses and composition patterns
- 07_copy_clone.py — Deep clone and shallow copy operations
- 08_detach_gradients.py — Detach tensors and manage gradients for training
- 09_nested.py — Nest TensorDict instances and mix with other containers

## TensorDict-Specific Features
- 10_dynamic_keys.py — Runtime key addition, removal, and conditional field handling
- 11_mapping_interface.py — MutableMapping methods (keys, values, items, update)
- 12_nested_dict_handling.py — Automatic dict wrapping and recursive nesting
- 13_flatten_keys.py — Key flattening with custom separators and namespace management
- 14_pytree_integration.py — Advanced PyTree operations and torch.compile compatibility