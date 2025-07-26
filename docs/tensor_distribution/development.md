# TensorDistribution Development Guide

This document outlines the design requirements and implementation patterns for [`TensorDistribution`](src/tensorcontainer/tensor_distribution/base.py:14) and its subclasses in the [`src/tensorcontainer/tensor_distribution/`](src/tensorcontainer/tensor_distribution/) module.

## Objective

The [`tensor_distribution`](src/tensorcontainer/tensor_distribution/) module provides [`torch.distributions.Distribution`](https://pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution) functionality that enables direct application of tensor operations to probability distributions. The module maintains complete signature compatibility with [`torch.distributions`](https://pytorch.org/docs/stable/distributions.html) while extending [`TensorContainer`](src/tensorcontainer/tensor_container.py:1) functionality through [`TensorAnnotated`](src/tensorcontainer/tensor_annotated.py) inheritance.

## Architecture Requirements

### Signature Compatibility

All classes in [`tensorcontainer.tensor_distribution`](src/tensorcontainer/tensor_distribution/) must maintain exact signature compatibility with their corresponding [`torch.distributions`](https://pytorch.org/docs/stable/distributions.html) classes. This compatibility is enforced through automated testing using [`tests/tensor_distribution/conftest.py::assert_init_signatures_match`](tests/tensor_distribution/conftest.py).

**Implementation Requirement**: When [`torch.distributions`](https://pytorch.org/docs/stable/distributions.html) classes lack proper type annotations for [`__init__`](src/tensorcontainer/tensor_distribution/base.py:65) parameters, implementers **must** consult the class docstring to determine correct type hints.

### Distribution Delegation Pattern

[`TensorDistribution`](src/tensorcontainer/tensor_distribution/base.py:14) subclasses **must not** implement distribution-specific logic. Instead, each subclass **must** implement a [`dist()`](src/tensorcontainer/tensor_distribution/base.py:143) method that constructs and returns the equivalent [`torch.distributions`](https://pytorch.org/docs/stable/distributions.html) object using the instance's parameters.

**Implementation Requirement**: The [`dist()`](src/tensorcontainer/tensor_distribution/base.py:143) method **must** return the raw [`torch.distributions`](https://pytorch.org/docs/stable/distributions.html) instance, not a wrapped one (e.g., with [`Independent`](https://pytorch.org/docs/stable/distributions.html#torch.distributions.independent.Independent)).

**Design Principle**: [`TensorDistribution`](src/tensorcontainer/tensor_distribution/base.py:14) serves as a parameter management wrapper around [`torch.distributions`](https://pytorch.org/docs/stable/distributions.html), delegating all distribution operations to the underlying implementation via [`self.dist()`](src/tensorcontainer/tensor_distribution/base.py:143) calls.

### Parameter Broadcasting Requirements

Many [`torch.distributions`](https://pytorch.org/docs/stable/distributions.html) constructors accept parameters of type `Union[Number, Tensor]` or any specialization of `Number` (e.g. `float`). However, [`TensorContainer`](src/tensorcontainer/tensor_container.py) and [`TensorDistribution`](src/tensorcontainer/tensor_distribution/base.py:14) can only process `Union[Tensor, TensorContainer]` objects and require all parameters to have compatible shapes for broadcasting.

**Implementation Rule**: When the constructor signature contains `Union[Number, Tensor]` or any specialization of `Number` parameters, implementations **must** use [`torch.distributions.utils.broadcast_all`](https://pytorch.org/docs/stable/distributions.html#torch.distributions.utils.broadcast_all) to:
1. Convert scalar numbers to tensors
2. Broadcast all parameters to a common shape

This preprocessing ensures proper shape and device management within the [`TensorAnnotated`](src/tensorcontainer/tensor_annotated.py) framework.

**Decision Criterion**: If the constructor signature does not contain `Union[Number, Tensor]` parameters, simpler parameter handling approaches should be preferred.

### Validation Strategy

[`TensorDistribution`](src/tensorcontainer/tensor_distribution/base.py:14) accepts a [`validate_args`](src/tensorcontainer/tensor_distribution/base.py:69) parameter during initialization and stores it as the [`_validate_args`](src/tensorcontainer/tensor_distribution/base.py:63) attribute of the base class. Subclasses must pass this value to the underlying [`torch.distributions`](https://pytorch.org/docs/stable/distributions.html) object (if the constructor supports it).

**Validation Policy**: Parameter validation for [`TensorDistribution`](src/tensorcontainer/tensor_distribution/base.py:14) subclasses is generally unnecessary because the [`TensorDistribution.__init__`](src/tensorcontainer/tensor_distribution/base.py:65) method constructs the underlying distribution once via [`self.dist()`](src/tensorcontainer/tensor_distribution/base.py:86), triggering parameter validation in the [`torch.distributions`](https://pytorch.org/docs/stable/distributions.html) implementation.

**Exception Handling**: Implementations should only raise validation errors when required parameters needed for device and shape inference are missing or invalid.

### Property Implementation Pattern

Following the [`torch.distributions.Distribution`](https://pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution) pattern, basic distribution properties are provided through the [`TensorDistribution`](src/tensorcontainer/tensor_distribution/base.py:14) base class via delegation to [`self.dist()`](src/tensorcontainer/tensor_distribution/base.py:143).

**Specialization Rule**: Distribution-specific properties **must** be implemented only in the corresponding subclass, maintaining the same delegation pattern to the underlying [`torch.distributions`](https://pytorch.org/docs/stable/distributions.html) object.

## Implementation Patterns

### Annotated Attribute Pattern

All tensor parameters must be declared as annotated class attributes to enable automatic transformation by [`TensorAnnotated`](src/tensorcontainer/tensor_annotated.py) operations (e.g., [`.to()`](src/tensorcontainer/tensor_annotated.py), [`.expand()`](src/tensorcontainer/tensor_annotated.py)).

**Example Pattern**:
```python
class TensorNormal(TensorDistribution):
    _loc: Tensor
    _scale: Tensor

    def __init__(self, loc: Tensor, scale: Tensor, validate_args: Optional[bool] = None):
        self._loc = loc
        self._scale = scale
        super().__init__(loc.shape, loc.device, validate_args)

    def dist(self) -> Distribution:
        return Normal(self._loc, self._scale, validate_args=self._validate_args)
```

Note: If parameters like `loc` and `scale` could be scalars in the constructor signature, apply the broadcasting rules described in the "Parameter Broadcasting" section before assignment to ensure proper tensor handling.

### Lazy Distribution Creation

The actual [`torch.distributions.Distribution`](https://pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution) instance is created on-demand through the [`dist()`](src/tensorcontainer/tensor_distribution/base.py:143) method. This lazy evaluation pattern enables efficient tensor operations without premature distribution instantiation.

### Reconstruction Pattern

The [`_unflatten_distribution()`](src/tensorcontainer/tensor_distribution/base.py:115) class method reconstructs distribution instances from serialized tensor and metadata attributes. This method is called by [`_init_from_reconstructed()`](src/tensorcontainer/tensor_distribution/base.py:90) during operations like [`.to()`](src/tensorcontainer/tensor_annotated.py) and [`.expand()`](src/tensorcontainer/tensor_annotated.py).

**Customization Requirement**: Subclasses with complex parameter relationships **must** override [`_unflatten_distribution()`](src/tensorcontainer/tensor_distribution/base.py:115) to implement appropriate reconstruction logic.

**Example Implementation**:
```python
@classmethod
def _unflatten_distribution(cls, attributes: Dict[str, Any]):
    """For TensorCategorical, extract _probs and _logits from attributes."""
    return cls(
        probs=attributes.get("_probs"),
        logits=attributes.get("_logits"),
        validate_args=attributes.get("_validate_args"),
    )
```
