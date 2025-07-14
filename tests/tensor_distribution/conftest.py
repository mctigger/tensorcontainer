import inspect
from typing import Type

import pytest
import torch
from torch.distributions import Distribution

from tensorcontainer.tensor_distribution import TensorDistribution


@pytest.fixture(autouse=True)
def deterministic_seed():
    torch.manual_seed(0)


@pytest.fixture(autouse=True)
def preserve_distributions_validation():
    """
    Preserve and restore torch.distributions validation state around each test.
    This prevents torch.compile usage in one test from affecting subsequent tests.

    This fixture automatically runs before and after every test in this directory,
    ensuring that any global state changes (like torch.compile disabling validation)
    don't leak between tests.
    """
    # Store the original validation state before the test
    original_validate_args = torch.distributions.Distribution._validate_args

    yield  # Run the test

    # Restore the original validation state after the test
    torch.distributions.Distribution.set_default_validate_args(original_validate_args)


@pytest.fixture(autouse=True)
def with_distributions_validation():
    """
    Fixture to ensure distributions validation is enabled for specific tests.

    Usage:
        def test_validation_required(self, with_distributions_validation):
            # Validation is guaranteed to be enabled in this test
            with pytest.raises(ValueError):
                TensorNormal(loc=torch.tensor([1.0]), scale=torch.tensor([-0.1]))
    """
    # Force enable validation for this test
    torch.distributions.Distribution.set_default_validate_args(True)
    yield


def normalize_device(dev: torch.device) -> torch.device:
    d = torch.device(dev)
    # If no index was given, fill in current_device() for CUDA, leave CPU as-is
    if d.type == "cuda" and d.index is None:
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()  # e.g. 0
            return torch.device(f"cuda:{idx}")
        else:
            # If CUDA is not available, return the device as-is
            return d
    return d


def assert_init_signatures_match(
    td_class: Type[TensorDistribution], torch_dist_class: Type[Distribution]
) -> None:
    """
    Assert that __init__ signatures match between TensorDistribution and Distribution.

    This function compares the __init__ signatures of a TensorDistribution subclass
    and a torch.distributions.Distribution subclass, asserting that they are
    identical, disregarding the 'self' and 'validate_args' parameters.

    Args:
        td_class (Type[TensorDistribution]): The TensorDistribution class.
        torch_dist_class (Type[Distribution]): The torch.distribution.Distribution class.

    """
    td_sig = inspect.signature(td_class.__init__)
    torch_sig = inspect.signature(torch_dist_class.__init__)

    td_params = [
        p.replace(annotation=inspect.Parameter.empty)
        for p in td_sig.parameters.values()
        if p.name != "self"
    ]
    torch_params = [
        p.replace(annotation=inspect.Parameter.empty)
        for p in torch_sig.parameters.values()
        if p.name not in ("self", "validate_args")
    ]

    td_sig_compare = td_sig.replace(parameters=td_params)
    torch_sig_compare = torch_sig.replace(parameters=torch_params)

    # Compare string representations of return annotations to handle __future__.annotations
    td_return_annotation_str = str(td_sig_compare.return_annotation)
    torch_return_annotation_str = str(torch_sig_compare.return_annotation)

    # Create new signatures with empty return annotations for comparison
    td_sig_no_return = td_sig_compare.replace(return_annotation=inspect.Signature.empty)
    torch_sig_no_return = torch_sig_compare.replace(return_annotation=inspect.Signature.empty)

    assert td_sig_no_return == torch_sig_no_return and td_return_annotation_str == torch_return_annotation_str, (
        f"__init__ signatures do not match between {td_class.__name__} and "
        f"{torch_dist_class.__name__}.\n"
        f"Got:      {td_class.__name__}{td_sig_compare}\n"
        f"Expected: {torch_dist_class.__name__}{torch_sig_compare}"
    )


def _get_torch_dist_properties(
    torch_dist_class: Type[Distribution],
) -> list[tuple[str, property]]:
    """
    Discover relevant public properties from a torch.distribution.Distribution class.
    This function inspects a torch.distribution.Distribution subclass and returns a
    list of its public properties, filtering out those that are defined in the base
    Distribution class to focus only on subclass-specific properties.
    Args:
        torch_dist_class (Type[Distribution]): The torch distribution class.
    Returns:
        A list of tuples, where each tuple contains the name and property object.
    """
    # Get all properties from the subclass
    subclass_properties = inspect.getmembers(
        torch_dist_class, predicate=lambda x: isinstance(x, property)
    )
    
    # Get all properties from the base Distribution class
    base_properties = inspect.getmembers(
        Distribution, predicate=lambda x: isinstance(x, property)
    )
    
    # Create a set of base class property names for efficient lookup
    base_property_names = {name for name, _ in base_properties}

    properties = []
    # Iterate over all public properties found in the torch.distribution subclass.
    for name, prop in subclass_properties:
        # Skip private properties, following the convention of an initial underscore.
        if name.startswith("_"):
            continue

        # Skip properties that are defined in the base Distribution class
        if name in base_property_names:
            continue
            
        properties.append((name, prop))
    return properties


def assert_properties_signatures_match(
    td_class: Type[TensorDistribution], torch_dist_class: Type[Distribution]
) -> None:
    """
    Asserts that public properties of a torch.distribution.Distribution are mirrored
    in the corresponding TensorDistribution subclass.

    This function verifies that for each public property in the torch distribution,
    there is a corresponding property in the TensorDistribution with a matching
    getter signature. This ensures that the TensorDistribution subclass maintains a
    consistent interface with the original torch distribution.

    Args:
        td_class (Type[TensorDistribution]): The TensorDistribution class to check.
        torch_dist_class (Type[Distribution]): The corresponding torch distribution class.
    """
    dist_properties = _get_torch_dist_properties(torch_dist_class)

    # Iterate over all public properties found in the torch.distribution class.
    for name, prop in dist_properties:
        # Check if the TensorDistribution class has the same property.
        assert hasattr(
            td_class, name
        ), f"{td_class.__name__} is missing property '{name}' from {torch_dist_class.__name__}"
        td_prop = getattr(td_class, name)
        assert isinstance(
            td_prop, property
        ), f"Attribute '{name}' in {td_class.__name__} is not a property"

        # If the torch distribution property has no getter, skip it.
        if prop.fget is None:
            continue

        # Access the getter method (fget) for both the TensorDistribution and torch.distribution properties.
        assert (
            td_prop.fget is not None
        ), f"Property '{name}' in {td_class.__name__} is missing a getter"

        # Retrieve the signatures of the getter methods using inspect.signature.
        td_getter_sig = inspect.signature(td_prop.fget)
        dist_getter_sig = inspect.signature(prop.fget)

        # Remove annotations from parameters for a less strict comparison, focusing on names and kinds.
        td_getter_params = [
            p.replace(annotation=inspect.Parameter.empty)
            for p in td_getter_sig.parameters.values()
        ]
        dist_getter_params = [
            p.replace(annotation=inspect.Parameter.empty)
            for p in dist_getter_sig.parameters.values()
        ]

        # Create comparable versions of the signatures, ignoring return annotations.
        td_getter_sig_compare = td_getter_sig.replace(
            parameters=td_getter_params,
            return_annotation=inspect.Signature.empty,
        )
        dist_getter_sig_compare = dist_getter_sig.replace(
            parameters=dist_getter_params,
            return_annotation=inspect.Signature.empty,
        )

        # Assert that the signatures of the getter methods match.
        assert td_getter_sig_compare == dist_getter_sig_compare, (
            f"Getter signature mismatch for property '{name}':\n"
            f"  {td_class.__name__}: {td_getter_sig}\n"
            f"  {torch_dist_class.__name__}: {dist_getter_sig}"
        )


def assert_property_values_match(td: TensorDistribution) -> None:
    """
    Asserts that property values match between a TensorDistribution and the
    torch.distribution.Distribution instance it wraps.

    This function iterates through the public properties of the underlying torch
    distribution, retrieves the value of each property from both the `td` instance
    and its corresponding `td.dist()` instance, and asserts that the values are
    identical. For tensor values, `torch.testing.assert_close` is used for robust
    comparison.

    Args:
        td (TensorDistribution): The TensorDistribution instance to check.
    """
    dist = td.dist()
    properties_to_check = _get_torch_dist_properties(dist.__class__)

    for name, _ in properties_to_check:
        td_value = getattr(td, name)
        dist_value = getattr(dist, name)

        if isinstance(td_value, torch.Tensor):
            try:
                torch.testing.assert_close(td_value, dist_value)
            except AssertionError as e:
                raise AssertionError(
                    f"Tensor value mismatch for property '{name}': {str(e)}"
                ) from e
        else:
            assert td_value == dist_value, (
                f"Value mismatch for property '{name}':\n"
                f"  {td.__class__.__name__}: {td_value}\n"
                f"  {dist.__class__.__name__}: {dist_value}"
            )
