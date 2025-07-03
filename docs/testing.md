# Testing Guide

Our testing philosophy is simple: **every piece of functionality should be verified by a clean, isolated, and readable test**. We use the `pytest` framework to write and run our tests.

This guide outlines our core principles for creating a high-quality and maintainable test suite.

---

## Core Principles

### Test File Structure
Every test file (e.g., `test_container.py`) must begin with a module-level docstring. This docstring should provide a short summary of the test classes contained within that file, giving a high-level overview of the test coverage.

### Test Class Structure
All tests must be organized into classes. Every test class must have a docstring that clearly explains its purpose. The docstring should follow this format:

1.  **Summary**: 1-2 sentences giving a high-level overview of the feature being tested.
2.  **Test List**: A bulleted list detailing the specific behaviors verified by the test suite.

For example:
```python
class TestTensorContainerInitialization:
    """
    Tests the initialization logic of the TensorContainer.

    This suite verifies that:
    - The container is created successfully with a valid tensor.
    - The 'requires_grad' flag is set correctly during instantiation.
    - A TypeError is raised when input data is not a PyTorch tensor.
    """
    # ... test methods go here
```

### Test Method Rules

- Isolate Everything: Each test method (def test_*) should verify only one specific behavior or outcome. Don't test multiple things at once.
- Parametrize, Don't Branch: We use pytest.mark.parametrize to test multiple scenarios of a single behavior. This helps avoid if/else logic inside the test itself, keeping its purpose clear and focused.
- When parametrizing a test, make sure the all the parameters are of similar type. For example the following is a good parameterization:

    ```
    @pytest.mark.parametrize(
            "original_shape,expected_elements",
            [
                ((2, 3), 6),
                ((4, 5), 20),
                ((1, 10), 10),
                ((3, 2, 2), 12),
            ],
        )
    ```
    And this is a bad test:

     ```
    @pytest.mark.parametrize(
            "shape_or_type,expected_elements_or_type",
            [
                ((2, 3), 6),
                (torch.float, torch.float),
                (torch.double, torch.double),
                ((3, 2, 2), 12),
            ],
        )
    ```


- Clear Naming: Test classes and methods should have descriptive names that immediately convey what they are testing.
- Do not overcomplicate: Instead of creating complex tests that are very general, create many simple test cases that check exactly one thing. 
- Assert only the properties that are relevant to the test case. Complicated asserts indicate a overly complex test. In that case, split the test into multiple.
- Add a small docstring that explains the reasoning behind the test:

    ```
    def test_optional_field_none(self):
        """
        The TensorDataClass should allow fields to be optional. 
        These fields are set later.
        """
    ```
- Use the context7 MCP when you need to access API of installed packages (such as torch, numpy, torchvision, ...)
- Use the testing utilities from torch.testing when working with tensors.
- Never change the implementation that is tested. For example if your task is to test TensorContainer.flatten() you are not allowed to make any changes in TensorContainer.