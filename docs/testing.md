# Testing Guidelines

This document outlines the best practices and conventions for writing tests in this project. Adhering to these guidelines ensures that our tests are consistent, maintainable, and effective.

## General Philosophy

Tests should be clear, concise, and focused. Each test should verify a specific piece of functionality. We heavily rely on `pytest` for its powerful features like parametrization and fixtures, and `torch.testing` for reliable tensor comparisons. A key requirement for this project is to ensure that our components are compatible with `torch.compile`, so most tests should cover both eager and compiled execution paths.

## Test File Structure

A typical test file is structured as follows:

1.  **Imports**: Standard library, third-party libraries (`pytest`, `torch`), and then internal source code imports.
2.  **Test Data Setup**:
    *   A dedicated `TensorDataClass` subclass for the specific tests (e.g., `SampleTensorDataClass`).
    *   Helper functions to create instances of this dataclass with various configurations (e.g., `create_sample_tdc`).
3.  **Test Class**: Tests for a specific feature (like `__getitem__` or `__setitem__`) are grouped within a class (e.g., `TestGetItem`).

## Core Pattern: The Central Verification Helper

To keep tests clean, consistent, and robust, we use a central helper method within each test class (e.g., `_run_and_verify_setitem`). This method encapsulates the entire test-and-verify logic for a given operation.

### Responsibilities of a Verification Helper

1.  **Define and Execute the Operation**: It defines a small, compilable function that performs the action under test (e.g., `tdc[idx] = value`).
2.  **Test `torch.compile` Compatibility**: It runs the operation through `torch.compile` and ensures there are no unexpected graph breaks. It includes a fallback mechanism (e.g., `fullgraph=False`) for operations that are not fully supported.
3.  **Compare Eager vs. Compiled Execution**: It uses a utility like `run_and_compare_compiled` to execute the operation in both eager and compiled modes, asserting that the outputs are identical.
4.  **Perform Comprehensive Assertions**: After execution, it verifies all critical properties of the result:
    *   **Type**: The output is of the expected `TensorDataClass` type.
    *   **Value Correctness**: The tensor values are correct using `torch.testing.assert_close`.
    *   **Shape**: The output's `shape` attribute is correct and consistent with its internal tensors.
    *   **Device**: The output is on the correct device.

### Example: Verification Helper for `__getitem__`

```python
# tests/tensor_dataclass/test_getitem.py

class TestGetItem:
    def _run_and_verify(self, tdc, idx, test_name):
        """
        Central helper method to encapsulate common testing logic for __getitem__.
        """
        # 1. Compile and run the operation
        def _get_item(tdc, idx):
            return tdc[idx]

        result, _ = run_and_compare_compiled(
            _get_item, tdc, idx, fullgraph=True, expected_graph_breaks=0
        )

        # 2. Assert type
        assert isinstance(result, SampleTensorDataClass)

        # 3. Calculate expected values (handling event dims)
        expected_labels = tdc.labels[idx]
        features_idx = ... # logic to add slicing for extra event dims
        expected_features = tdc.features[features_idx]

        # 4. Assert correctness
        torch.testing.assert_close(result.features, expected_features)
        torch.testing.assert_close(result.labels, expected_labels)

        # 5. Assert metadata
        assert result.device == expected_features.device
        assert result.shape + ... == expected_labels.shape
```

### Test Execution Flow

This diagram shows the flow of a single test case using the central helper pattern.

```mermaid
graph TD
    A[Test Function e.g., test_getitem_with_slice(...)] -- calls --> B(Central Helper: _run_and_verify);
    B -- 1. Defines --> C{Operation: _get_item(tdc, idx)};
    B -- 2. Compiles & Runs --> D[run_and_compare_compiled(op)];
    B -- 3. Asserts --> E[Type Check];
    B -- 3. Asserts --> F[Value Check (torch.testing.assert_close)];
    B -- 3. Asserts --> G[Shape & Device Check];
```

## Writing Test Cases: A Step-by-Step Guide

Follow these steps to write effective and maintainable tests.

### 1. Categorize Tests by Behavior

Instead of creating one large test method for a feature, **DO** create multiple, focused test methods that target a specific behavior. This makes tests easier to understand, debug, and maintain.

**DON'T** mix different test scenarios (e.g., assigning a scalar, a tensor, and a `TensorDataClass`) in the same test method.

```python
# tests/tensor_dataclass/test_setitem.py

class TestSetItem:
    # GOOD: One test for assigning a scalar
    def test_setitem_basic_indexing_with_scalar(self, test_name, idx):
        # ...

    # GOOD: Another test for assigning a TensorDataClass
    def test_setitem_basic_indexing_with_tdc(self, test_name, idx):
        # ...
```

### 2. Use Docstrings as Mini-Specs

**DO** write a clear, multi-line docstring for every test method. The docstring should:
1.  Explain the purpose of the test in one sentence.
2.  Provide a concise, conceptual example using a standard `torch.Tensor` to illustrate the behavior being tested. This serves as a mini-specification.

This practice makes the test's intent immediately obvious without needing to read its implementation.

```python
# tests/tensor_dataclass/test_setitem.py

def test_setitem_advanced_indexing_with_scalar(self, test_name, idx):
    """Tests advanced indexing with a scalar value.

    This test covers advanced indexing using lists of integers or tensors.
    It checks that assigning a scalar value to elements specified by
    advanced indexing works correctly and compiles.

    Example with torch.Tensor:
        >>> tensor = torch.zeros(10)
        >>> indices = [0, 4, 8]
        >>> tensor[indices] = 1.0
        >>> # tensor is now [1., 0., 0., 0., 1., 0., 0., 0., 1., 0.]
    """
    tdc_initial = create_sample_tdc()
    value = 0.0
    self._run_and_verify_setitem(tdc_initial, idx, value, test_name)
```

### 3. Parametrize with Focused Groups

**DO** group related test cases into focused lists for parametrization. This is more readable and organized than a single, monolithic list of parameters. Use descriptive names for these lists (e.g., `BASIC_INDEXING_CASES`, `ADVANCED_INDEXING_CASES`, `BOOLEAN_MASK_CASES`).

**DO** provide descriptive `ids` for `pytest.mark.parametrize` to get readable test reports.

```python
# tests/tensor_dataclass/test_setitem.py

# GOOD: Focused groups of test cases
BASIC_INDEXING_CASES = [
    ("int", 5),
    ("slice", slice(2, 15)),
]

ADVANCED_INDEXING_CASES = [
    ("list_int", ([0, 4, 2, 19, 7])),
    ("long_tensor", (torch.tensor([0, 4, 2, 19, 7]))),
]

class TestSetItem:
    @pytest.mark.parametrize("test_name,idx", BASIC_INDEXING_CASES, ids=[c[0] for c in BASIC_INDEXING_CASES])
    def test_setitem_basic_indexing_with_scalar(self, test_name, idx):
        # ...

    @pytest.mark.parametrize("test_name,idx", ADVANCED_INDEXING_CASES, ids=[c[0] for c in ADVANCED_INDEXING_CASES])
    def test_setitem_advanced_indexing_with_scalar(self, test_name, idx):
        # ...
```

### 4. Test for Errors Systematically

**DO** test for expected errors to ensure the code fails predictably.

1.  **Group Invalid Cases**: Just like valid cases, group invalid test cases into lists (e.g., `INVALID_INDEXING_CASES`).
2.  **Use `pytest.raises`**: Use `pytest.raises` as a context manager to assert that a specific exception is raised.
3.  **Match the Error Message**: Use the `match` argument to check that the error message contains a specific, expected substring or matches a regex pattern. This prevents tests from passing on the correct exception type but the wrong error condition.

```python
# tests/tensor_dataclass/test_setitem.py

INVALID_INDEXING_CASES = [
    (
        "shape_mismatch_tdc",
        (slice(None),),
        (19, 5), # Shape of the value TDC
        ValueError,
        r"Assignment failed for leaf at path.*", # Regex for the message
    ),
    ("index_out_of_bounds", 20, None, IndexError, "out of bounds"),
]

@pytest.mark.parametrize(
    "test_name,idx,value_shape,error,match",
    INVALID_INDEXING_CASES,
    ids=[case[0] for case in INVALID_INDEXING_CASES],
)
def test_setitem_invalid_inputs_raise_errors(self, test_name, idx, value_shape, error, match):
    tdc = create_sample_tdc()
    value_to_assign = ... # Setup the value to assign
    with pytest.raises(error, match=match): # DO: Use pytest.raises with match
        tdc[idx] = value_to_assign
```

**DON'T** use a generic `try...except` block, as this can hide bugs and makes the test's intent less clear.

## Golden Rules and Best Practices

*   **DO** use the **Central Verification Helper** pattern for all test classes.
*   **DO** categorize tests into separate methods based on behavior.
*   **DO** write a docstring with a `torch.Tensor` example for every test method.
*   **DO** use `torch.testing.assert_close` for comparing all tensors.
*   **DO** use `pytest.raises(ErrorType, match="...")` to test for specific errors.
*   **DON'T** put unrelated test cases into a single parametrization list.
*   **DON'T** use `assert (tensor1 == tensor2).all()`. It is not safe for floating-point comparisons.