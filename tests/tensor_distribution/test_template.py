"""
Template for testing distribution-specific properties of tensor distributions.

This template focuses EXCLUSIVELY on distribution-specific properties and should NOT
include tests for common methods inherited from TensorDistribution/TensorDataClass.

CUSTOMIZATION INSTRUCTIONS:
1. Replace `[DISTRIBUTION_NAME]` with your distribution's class name (e.g., Normal, Categorical).
2. Replace `[DISTRIBUTION_FILE_NAME]` with your distribution's file name (e.g., normal, categorical).
3. Replace `[TORCH_DISTRIBUTION_NAME]` with the corresponding `torch.distributions` class name.
4. Replace `param1`, `param2`, etc., with your distribution's specific parameter names (e.g., `loc`, `scale`).
5. Update parameter generation logic in `TEST_CASES` and helper functions.
6. Customize the parameter properties tests for your distribution's unique attributes.
7. Adapt the reference comparison tests to use the correct `torch.distributions` equivalent.
8. Add any other distribution-specific tests (e.g., validation of parameter constraints,
   special methods unique to your distribution).
"""

import pytest
import torch

# TODO: Uncomment and replace distribution_file_placeholder and TensorDistributionPlaceholder
# from tensorcontainer.tensor_distribution.distribution_file_placeholder import TensorDistributionPlaceholder


# TODO: Define TEST_CASES specific to your distribution's parameters and shapes.
# Example for a distribution with two parameters (e.g., loc, scale or concentration1, concentration0)
TEST_CASES = [
    # (batch_shape, param1_shape, param2_shape, reinterpreted_batch_ndims)
    ((), (1,), (1,), 0),  # Scalar distribution
    ((3,), (3,), (3,), 0),  # 1D batch shape
    ((2, 4), (2, 4), (2, 4), 0),  # 2D batch shape
    ((2, 3), (2, 3), (2, 3), 1),  # Reinterpreted batch dim
    ((2, 3, 4), (2, 3, 4), (2, 3, 4), 2),  # Multiple reinterpreted batch dims
]


# TODO: Add helper functions for generating parameters if needed.
# Example:
def _generate_params(batch_shape, param1_shape, param2_shape, device):
    # Customize this function to generate valid parameters for your distribution
    # For example, for Normal:
    param1 = torch.randn(batch_shape + param1_shape, device=device)
    param2 = (
        torch.rand(batch_shape + param2_shape, device=device) + 0.1
    )  # Ensure positive
    return param1, param2


class TestTensorDistributionNameInitialization:  # TODO: Replace DistributionName with your distribution's class name
    """
    Tests the initialization logic and parameter properties of the Tensor[DISTRIBUTION_NAME] distribution.
    """

    @pytest.mark.parametrize(
        "batch_shape, param1_shape, param2_shape, reinterpreted_batch_ndims",
        TEST_CASES,
        ids=[
            f"batch={bs},p1={p1s},p2={p2s},rbn={rbn}"
            for bs, p1s, p2s, rbn in TEST_CASES
        ],
    )
    def test_valid_initialization(
        self, batch_shape, param1_shape, param2_shape, reinterpreted_batch_ndims
    ):
        """
        Tests that Tensor[DISTRIBUTION_NAME] can be instantiated with valid parameters.
        """
        # TODO: Customize parameter generation for your distribution
        param1_val, param2_val = _generate_params(
            batch_shape, param1_shape, param2_shape, "cpu"
        )

        # TODO: Replace TensorDistributionPlaceholder with your actual TensorDistribution class
        # dist = TensorDistributionPlaceholder(
        #     param1=param1_val, # TODO: Replace param1 with your actual parameter name
        #     param2=param2_val, # TODO: Replace param2 with your actual parameter name
        #     reinterpreted_batch_ndims=reinterpreted_batch_ndims,
        #     shape=param1_val.shape, # Assuming shape is derived from param1
        #     device=param1_val.device,
        # )
        # assert isinstance(dist, TensorDistributionPlaceholder) # TODO: Replace TensorDistributionPlaceholder
        # TODO: Add assertions for specific parameter properties
        # assert_close(dist.param1, param1_val) # TODO: Replace param1
        # assert_close(dist.param2, param2_val) # TODO: Replace param2
        pass  # Placeholder to avoid empty test

    # TODO: Add tests for parameter validation (e.g., non-positive values, shape mismatches)
    # Example:
    # @pytest.mark.parametrize(
    #     "param1, param2",
    #     [
    #         (torch.tensor([-0.1]), torch.tensor([1.0])), # Invalid param1
    #         (torch.tensor([1.0]), torch.tensor([0.0])),  # Invalid param2
    #     ],
    # )
    # def test_invalid_parameter_values_raises_error(self, param1, param2):
    #     with pytest.raises(ValueError): # Or RuntimeError, depending on the error type
    #         # TODO: Replace TensorDistributionPlaceholder with your actual TensorDistribution class
    #         TensorDistributionPlaceholder(param1=param1, param2=param2, shape=param1.shape, device=param1.device)

    # TODO: Add tests for parameter conversion (e.g., logits <-> probs for Categorical/Bernoulli)
    # Example for Bernoulli/Categorical:
    # def test_logits_probs_conversion(self):
    #     logits = torch.randn(5)
    #     # TODO: Replace TensorDistributionPlaceholder with your actual TensorDistribution class
    #     dist_logits = TensorDistributionPlaceholder(_logits=logits, shape=logits.shape, device=logits.device)
    #     assert_close(dist_logits.probs, torch.sigmoid(logits))
    #     probs = torch.rand(5)
    #     # TODO: Replace TensorDistributionPlaceholder with your actual TensorDistribution class
    #     dist_probs = TensorDistributionPlaceholder(_probs=probs, shape=probs.shape, device=probs.device)
    #     assert_close(dist_probs.logits, torch.log(probs / (1 - probs + 1e-8)))


class TestTensorDistributionNameReferenceComparison:  # TODO: Replace DistributionName
    """
    Tests that Tensor[DISTRIBUTION_NAME] behaves consistently with torch.distributions.[TORCH_DISTRIBUTION_NAME].
    """

    @pytest.mark.parametrize(
        "batch_shape, param1_shape, param2_shape, reinterpreted_batch_ndims",
        TEST_CASES,
        ids=[
            f"batch={bs},p1={p1s},p2={p2s},rbn={rbn}"
            for bs, p1s, p2s, rbn in TEST_CASES
        ],
    )
    def test_dist_property_and_compilation(
        self, batch_shape, param1_shape, param2_shape, reinterpreted_batch_ndims
    ):
        """
        Tests the .dist() property and its compatibility with torch.compile.
        """
        param1_val, param2_val = _generate_params(
            batch_shape, param1_shape, param2_shape, "cpu"
        )

        # TODO: Replace TensorDistributionPlaceholder with your actual TensorDistribution class
        # td_dist = TensorDistributionPlaceholder(
        #     param1=param1_val, # TODO: Replace param1
        #     param2=param2_val, # TODO: Replace param2
        #     reinterpreted_batch_ndims=reinterpreted_batch_ndims,
        #     shape=param1_val.shape,
        #     device=param1_val.device,
        # )

        # Test .dist() property
        # torch_dist = td_dist.dist()
        # assert isinstance(torch_dist, Independent)
        # assert isinstance(torch_dist.base_dist, TorchDistributionPlaceholder) # TODO: Replace TorchDistributionPlaceholder
        # TODO: Add assertions for batch_shape and event_shape if applicable
        # assert torch_dist.batch_shape == expected_batch_shape
        # assert torch_dist.event_shape == expected_event_shape

        # Test compilation of .dist()
        # def get_dist(td):
        #     return td.dist()

        # compiled_torch_dist, _ = run_and_compare_compiled(get_dist, td_dist, fullgraph=False)
        # assert isinstance(compiled_torch_dist, Independent)
        # assert isinstance(compiled_torch_dist.base_dist, TorchDistributionPlaceholder) # TODO: Replace TorchDistributionPlaceholder
        # TODO: Add assertions for compiled distribution's properties if needed
        pass  # Placeholder to avoid empty test

    # TODO: Add tests for specific distribution methods that have direct torch.distributions equivalents
    # and are NOT covered by base TensorDistribution tests (e.g., cdf, icdf, enumerate_support).
    # Example for cdf:
    # @pytest.mark.parametrize(
    #     "batch_shape, param1_shape, param2_shape, reinterpreted_batch_ndims",
    #     TEST_CASES,
    #     ids=[
    #         f"batch={bs},p1={p1s},p2={p2s},rbn={rbn}"
    #         for bs, p1s, p2s, rbn in TEST_CASES
    #     ],
    # )
    # def test_cdf_matches_torch_distribution(
    #     self, batch_shape, param1_shape, param2_shape, reinterpreted_batch_ndims
    # ):
    #     param1_val, param2_val = _generate_params(batch_shape, param1_shape, param2_shape, "cpu")
    #     # TODO: Replace TensorDistributionPlaceholder with your actual TensorDistribution class
    #     td_dist = TensorDistributionPlaceholder(
    #         param1=param1_val,
    #         param2=param2_val,
    #         reinterpreted_batch_ndims=reinterpreted_batch_ndims,
    #         shape=param1_val.shape,
    #         device=param1_val.device,
    #     )
    #     value = td_dist.sample() # Or generate a specific value
    #     assert_close(td_dist.cdf(value), td_dist.dist().cdf(value))

    # TODO: Add tests for compilation of distribution-specific methods
    # Example for cdf compilation:
    # @pytest.mark.parametrize(
    #     "batch_shape, param1_shape, param2_shape, reinterpreted_batch_ndims",
    #     TEST_CASES,
    #     ids=[
    #         f"batch={bs},p1={p1s},p2={p2s},rbn={rbn}"
    #         for bs, p1s, p2s, rbn in TEST_CASES
    #     ],
    # )
    # def test_cdf_compilation(
    #     self, batch_shape, param1_shape, param2_shape, reinterpreted_batch_ndims
    # ):
    #     param1_val, param2_val = _generate_params(batch_shape, param1_shape, param2_shape, "cpu")
    #     # TODO: Replace TensorDistributionPlaceholder with your actual TensorDistribution class
    #     td_dist = TensorDistributionPlaceholder(
    #         param1=param1_val,
    #         param2=param2_val,
    #         reinterpreted_batch_ndims=reinterpreted_batch_ndims,
    #         shape=param1_val.shape,
    #         device=param1_val.device,
    #     )
    #     value = td_dist.sample()
    #     def cdf_fn(dist, val):
    #         return dist.cdf(val)
    #     eager_cdf, compiled_cdf = run_and_compare_compiled(cdf_fn, td_dist, value, fullgraph=False)
    #     assert_close(eager_cdf, compiled_cdf)
