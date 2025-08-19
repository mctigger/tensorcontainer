# TensorDataClass Examples

This folder contains small, runnable scripts that demonstrate how to use TensorDataClass for common workflows. Each script can be executed directly:

- Basic usage: `python examples/tensor_dataclass/01_basic.py`

Index:
- 01_basic.py — Define a schema, instantiate with shape/device, and access fields
- 02_indexing.py — Index and slice batch dimensions
- 03_shape_ops.py — Reshape, unsqueeze, and squeeze batch dimensions
- 04_stack_cat.py — Combine instances with torch.stack and torch.cat
- 05_device.py — Move instances between CPU and CUDA (if available)
- 06_nested_inheritance.py — Nesting and inheritance patterns
- 07_copy_clone.py — Shallow copy vs deep clone

