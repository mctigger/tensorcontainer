# TensorDataClass User Guide

`TensorDataClass` is a dataclass-based container for PyTorch tensors, designed to provide strong typing, IDE support, and structured data management for tensor operations. It automatically converts annotated Python classes into dataclasses, facilitating the definition of fixed schemas for heterogeneous tensor data.

## 1. Quick Introduction

This section provides a brief overview of `TensorDataClass` and its core functionality through a practical example.

**Getting Started Example:**

```python
import torch
from tensorcontainer import TensorDataClass

class MyData(TensorDataClass):
    features: torch.Tensor
    labels: torch.Tensor

# Create an instance with batch shape (4,)
data = MyData(
    features=torch.randn(4, 10),
    labels=torch.arange(4).float(),
    shape=(4,),
    device="cpu"
)

print(data.features.shape)  # Output: torch.Size([4, 10])
# Access fields directly
data.features = torch.ones(4, 10) # Type-checked assignment
```

## 2. Intended Use

`TensorDataClass` is designed for scenarios where collections of tensors share a common batch dimension and have a predefined, fixed structure. It addresses the organization and access of heterogeneous tensor data in a type-safe and readable manner, particularly in domains such as reinforcement learning, batch processing, or applications involving structured data where each field is a tensor. It enforces consistent batch semantics across all contained tensors, ensuring that operations like slicing, device transfer, and shape manipulation apply uniformly.

## 3. Use Cases

*   **Reinforcement Learning Data:** Representing observations, actions, rewards, and next observations in a clear, type-safe structure for training agents.
    ```python
    class ExperienceBatch(TensorDataClass):
        observations: torch.Tensor  # (batch_size, obs_dim)
        actions: torch.Tensor       # (batch_size, action_dim)
        rewards: torch.Tensor       # (batch_size,)
        dones: torch.Tensor         # (batch_size,)
        next_observations: torch.Tensor # (batch_size, obs_dim)
    ```
*   **Batched Sensor Readings:** Grouping different sensor outputs (e.g., image, lidar, IMU data) from a batch of samples.
    ```python
    class SensorBatch(TensorDataClass):
        image: torch.Tensor  # (batch_size, channels, height, width)
        lidar: torch.Tensor  # (batch_size, num_points, num_features)
        imu: torch.Tensor    # (batch_size, 6) # e.g., 3-axis accel, 3-axis gyro
    
    # Example usage:
    sensor_data = SensorBatch(
        image=torch.randn(16, 3, 64, 64),
        lidar=torch.randn(16, 1024, 3),
        imu=torch.randn(16, 6),
        shape=(16,)
    )
    print(sensor_data.image.shape) # torch.Size([16, 3, 64, 64])
    ```
*   **Model Inputs/Outputs:** Defining the expected input or output structure for neural networks, ensuring type consistency and simplifying data passing.
    ```python
    class ModelInput(TensorDataClass):
        input_features: torch.Tensor # (batch_size, input_dim)
        attention_mask: torch.Tensor # (batch_size, seq_len)
    
    class ModelOutput(TensorDataClass):
        logits: torch.Tensor         # (batch_size, num_classes)
        hidden_states: torch.Tensor  # (batch_size, seq_len, hidden_dim)
    
    # Example usage:
    input_data = ModelInput(
        input_features=torch.randn(8, 128),
        attention_mask=torch.ones(8, 32).bool(),
        shape=(8,)
    )
    # Assuming a model that takes ModelInput and returns ModelOutput
    # output_data = model(input_data)
    # print(output_data.logits.shape)
    ```
*   **Structured Datasets:** Handling datasets where each entry consists of multiple tensor components (e.g., a dataset of images and their corresponding labels, bounding boxes, and masks).
*   **Nested Data Structures:** `TensorDataClass` supports nesting, allowing for complex hierarchical data representations.
    ```python
    class InnerData(TensorDataClass):
        c: torch.Tensor

    class OuterData(TensorDataClass):
        a: torch.Tensor
        inner: InnerData
    ```

## 4. Limitations

*   **Fixed Schema:** Unlike `TensorDict`, `TensorDataClass` does not support dynamic addition or removal of fields at runtime. Its structure is defined at compile-time through class annotations.
*   **No `eq=True`:** `TensorDataClass` instances cannot be compared for equality using `==` (i.e., `eq=False` is enforced). This is due to the complexities and potential performance issues of comparing floating-point tensors for exact equality.
*   **Not for Arbitrary Key-Value Storage:** It's not intended as a general-purpose dictionary for tensors where keys are arbitrary and can change.

## 5. Key Design Decisions

*   **Automatic Dataclass Generation:** Subclasses of `TensorDataClass` are automatically converted into Python dataclasses. This provides:
    *   **Field-based Access:** `obj.field` syntax for intuitive access.
    *   **Static Typing & IDE Support:** Enables type checking, autocomplete, and better code readability.
    *   **Natural Inheritance:** Fields are merged across the inheritance hierarchy, promoting code reuse.
*   **`eq=False` Enforcement:** Equality comparison is disabled (`eq=False`) to prevent issues with tensor comparisons (e.g., floating-point precision).
*   **PyTree Integration:** `TensorDataClass` instances are automatically registered as PyTrees, allowing seamless integration with `torch.utils._pytree` functions and enabling efficient operations like `torch.stack` and `torch.cat` while preserving the data structure. Tensor fields become PyTree leaves, while non-tensor fields are treated as metadata.
*   **`torch.compile` Compatibility:** Designed with `torch.compile` in mind, ensuring static structure, efficient attribute access, and safe copying mechanisms to avoid graph breaks.
*   **Batch and Event Dimensions:** It enforces consistent batch dimensions across all tensor fields, while allowing event dimensions to vary. This is crucial for maintaining batch semantics during tensor operations.
*   **Device and Shape Validation:** During initialization, it validates that all tensor fields have compatible batch shapes and reside on compatible devices, providing detailed error messages for debugging.

## 6. Comparison with TensorDict

| Feature           | TensorDataClass                               | TensorDict                                    |
| :---------------- | :-------------------------------------------- | :-------------------------------------------- |
| **Access Pattern** | `obj.field` (attribute access)                | `obj["key"]` (dictionary-style access)        |
| **Type Safety**   | Static typing, IDE autocomplete, compile-time checks | Runtime checks, less IDE support for keys     |
| **IDE Support**   | Full autocomplete, type hints                 | Limited (keys are strings, not attributes)    |
| **Memory Usage**  | Lower (`slots=True` by default)               | Higher (uses a dictionary for storage)        |
| **Field Definition** | Compile-time (defined in class)               | Runtime (can add/remove keys dynamically)     |
| **Inheritance**   | Natural OOP inheritance patterns              | Composition (can contain other `TensorDict`s) |
| **Dynamic Fields** | Not supported                                 | Full support                                  |
| **Use Case**      | Fixed, well-defined data schemas               | Flexible, dynamic key-value tensor storage    |

**When to choose `TensorDataClass`:**

*   When your data has a fixed, known structure (schema) that won't change at runtime.
*   When type safety, IDE autocomplete, and static analysis are high priorities for code maintainability and readability.
*   When you want to leverage Python's dataclass features and OOP inheritance for your tensor containers.
*   When memory efficiency and faster attribute access are critical.

**When to choose `TensorDict`:**

*   When you need a flexible container where the keys (and thus the contained tensors) can be added, removed, or changed dynamically at runtime.
*   When the structure of your data is not strictly fixed or can vary.
*   When you prefer a dictionary-like interface for managing your tensors.
*   When you need to store arbitrary key-value pairs where values are tensors or other `TensorDict`s.