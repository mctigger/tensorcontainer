# TensorDataClass Examples

This folder contains small, runnable scripts that demonstrate how to use TensorDataClass for common workflows. Each script can be executed directly:

- Basic usage: `python examples/tensor_dataclass/01_basic.py`

Index:
- 01_basic.py — Define a schema, instantiate with shape/device, and access fields
- 02_indexing.py — Index and slice batch dimensions
- 03_shape_ops.py — Reshape batch dimensions
- 04_stack.py — Combine instances with torch.stack
- 05_device.py — Move instances between CPU and CUDA (if available)
- 06_inheritance.py — Inheritance patterns
- 07_copy_clone.py — Deep clone and copy operations
- 08_detach_gradients.py — Detach tensors and manage gradients

