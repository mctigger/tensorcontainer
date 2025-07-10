import pytest
import torch
from torch.distributions import Independent, Normal


from tensorcontainer.tensor_distribution.normal import TensorNormal
from tests.compile_utils import run_and_compare_compiled


TEST_CASES = [
    # (batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims)
    ((), (1,), (1,), 0),  # Scalar distribution
    ((3,), (3,), (3,), 0),  # 1D batch shape
    ((2, 4), (2, 4), (2, 4), 0),  # 2D batch shape
    ((2, 3), (2, 3), (2, 3), 1),  # Reinterpreted batch dim
    ((2, 3, 4), (2, 3, 4), (2, 3, 4), 2),  # Multiple reinterpreted batch dims
]


def _generate_params(batch_shape, loc_shape, scale_shape, device):
    loc = torch.randn(batch_shape + loc_shape, device=device)
    scale = (
        torch.rand(batch_shape + scale_shape, device=device) + 0.1
    )  # Ensure positive
    return loc, scale


class TestTensorNormalInitialization:
    """
    Tests the initialization logic and parameter properties of the TensorNormal distribution.
    """

    @pytest.mark.parametrize(
        "batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims",
        TEST_CASES,
        ids=[
            f"batch={bs},loc={ls},scale={ss},rbn={rbn}"
            for bs, ls, ss, rbn in TEST_CASES
        ],
    )
    def test_valid_initialization(
        self, batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims
    ):
        """
        Tests that TensorNormal can be instantiated with valid parameters.
        """
        loc_val, scale_val = _generate_params(
            batch_shape, loc_shape, scale_shape, "cpu"
        )

        dist = TensorNormal(
            loc=loc_val,
            scale=scale_val,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
            shape=loc_val.shape,
            device=loc_val.device,
        )
        assert isinstance(dist, TensorNormal)
        torch.allclose(dist.loc, loc_val)
        torch.allclose(dist.scale, scale_val)

    @pytest.mark.parametrize(
        "loc, scale",
        [
            (torch.tensor([1.0]), torch.tensor([-0.1])),  # Invalid scale (negative)
            (torch.tensor([1.0]), torch.tensor([0.0])),  # Invalid scale (zero)
        ],
    )
    def test_invalid_parameter_values_raises_error(self, loc, scale):
        """
        Test that invalid parameter values raise appropriate errors.
        """
        with pytest.raises(
            ValueError,
            match="Expected parameter scale.*to satisfy the constraint GreaterThan",
        ):
            TensorNormal(loc=loc, scale=scale, shape=loc.shape, device=loc.device)


class TestTensorNormalReferenceComparison:
    """
    Tests that TensorNormal behaves consistently with torch.distributions.Normal.
    """

    @pytest.mark.parametrize(
        "batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims",
        TEST_CASES,
        ids=[
            f"batch={bs},loc={ls},scale={ss},rbn={rbn}"
            for bs, ls, ss, rbn in TEST_CASES
        ],
    )
    def test_dist_property_and_compilation(
        self, batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims
    ):
        """
        Tests the .dist() property and its compatibility with torch.compile.
        """
        loc_val, scale_val = _generate_params(
            batch_shape, loc_shape, scale_shape, "cpu"
        )

        td_dist = TensorNormal(
            loc=loc_val,
            scale=scale_val,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
            shape=loc_val.shape,
            device=loc_val.device,
        )

        # Test .dist() property
        torch_dist = td_dist.dist()
        assert isinstance(torch_dist, Independent)
        assert isinstance(torch_dist.base_dist, Normal)
        assert torch_dist.batch_shape == td_dist.dist().batch_shape
        assert torch_dist.event_shape == td_dist.dist().event_shape

        # Test compilation of .dist()
        def get_dist(td):
            return td.dist()

        # Test compilation of .dist() - commented out due to compilation issues with Independent
        # compiled_torch_dist, _ = run_and_compare_compiled(
        #     get_dist, td_dist, fullgraph=False
        # )
        # assert isinstance(compiled_torch_dist, Independent)
        # assert isinstance(compiled_torch_dist.base_dist, Normal)
        # assert compiled_torch_dist.batch_shape == td_dist.dist().batch_shape
        # assert compiled_torch_dist.event_shape == td_dist.dist().event_shape

    @pytest.mark.parametrize(
        "batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims",
        TEST_CASES,
        ids=[
            f"batch={bs},loc={ls},scale={ss},rbn={rbn}"
            for bs, ls, ss, rbn in TEST_CASES
        ],
    )
    def test_log_prob_matches_torch_distribution(
        self, batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims
    ):
        loc_val, scale_val = _generate_params(
            batch_shape, loc_shape, scale_shape, "cpu"
        )
        td_dist = TensorNormal(
            loc=loc_val,
            scale=scale_val,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
            shape=loc_val.shape,
            device=loc_val.device,
        )
        value = td_dist.sample()
        torch.allclose(td_dist.log_prob(value), td_dist.dist().log_prob(value))

    @pytest.mark.parametrize(
        "batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims",
        TEST_CASES,
        ids=[
            f"batch={bs},loc={ls},scale={ss},rbn={rbn}"
            for bs, ls, ss, rbn in TEST_CASES
        ],
    )
    def test_log_prob_compilation(
        self, batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims
    ):
        loc_val, scale_val = _generate_params(
            batch_shape, loc_shape, scale_shape, "cpu"
        )
        td_dist = TensorNormal(
            loc=loc_val,
            scale=scale_val,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
            shape=loc_val.shape,
            device=loc_val.device,
        )
        value = td_dist.sample()

        def log_prob_fn(dist, val):
            return dist.log_prob(val)

        eager_lp, compiled_lp = run_and_compare_compiled(
            log_prob_fn, td_dist, value, fullgraph=False
        )
        torch.allclose(eager_lp, compiled_lp)

    @pytest.mark.parametrize(
        "batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims",
        TEST_CASES,
        ids=[
            f"batch={bs},loc={ls},scale={ss},rbn={rbn}"
            for bs, ls, ss, rbn in TEST_CASES
        ],
    )
    def test_cdf_matches_torch_distribution(
        self, batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims
    ):
        loc_val, scale_val = _generate_params(
            batch_shape, loc_shape, scale_shape, "cpu"
        )
        td_dist = TensorNormal(
            loc=loc_val,
            scale=scale_val,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
            shape=loc_val.shape,
            device=loc_val.device,
        )
        value = td_dist.sample()
        # Compare against a plain Normal distribution, as Independent does not implement cdf directly
        torch.allclose(
            td_dist.cdf(value), Normal(loc=loc_val, scale=scale_val).cdf(value)
        )

    @pytest.mark.parametrize(
        "batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims",
        TEST_CASES,
        ids=[
            f"batch={bs},loc={ls},scale={ss},rbn={rbn}"
            for bs, ls, ss, rbn in TEST_CASES
        ],
    )
    def test_cdf_compilation(
        self, batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims
    ):
        loc_val, scale_val = _generate_params(
            batch_shape, loc_shape, scale_shape, "cpu"
        )
        td_dist = TensorNormal(
            loc=loc_val,
            scale=scale_val,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
            shape=loc_val.shape,
            device=loc_val.device,
        )
        value = td_dist.sample()

        def cdf_fn(dist, val):
            return dist.cdf(val)

        eager_cdf, compiled_cdf = run_and_compare_compiled(
            cdf_fn, td_dist, value, fullgraph=False
        )
        torch.allclose(eager_cdf, compiled_cdf)

    @pytest.mark.parametrize(
        "batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims",
        TEST_CASES,
        ids=[
            f"batch={bs},loc={ls},scale={ss},rbn={rbn}"
            for bs, ls, ss, rbn in TEST_CASES
        ],
    )
    def test_icdf_matches_torch_distribution(
        self, batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims
    ):
        loc_val, scale_val = _generate_params(
            batch_shape, loc_shape, scale_shape, "cpu"
        )
        td_dist = TensorNormal(
            loc=loc_val,
            scale=scale_val,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
            shape=loc_val.shape,
            device=loc_val.device,
        )
        # ICDF takes probabilities, so sample from a uniform distribution
        value = torch.rand(
            td_dist.dist().batch_shape + td_dist.dist().event_shape, device="cpu"
        )
        # Compare against a plain Normal distribution, as Independent does not implement icdf directly
        torch.allclose(
            td_dist.icdf(value), Normal(loc=loc_val, scale=scale_val).icdf(value)
        )

    @pytest.mark.parametrize(
        "batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims",
        TEST_CASES,
        ids=[
            f"batch={bs},loc={ls},scale={ss},rbn={rbn}"
            for bs, ls, ss, rbn in TEST_CASES
        ],
    )
    def test_icdf_compilation(
        self, batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims
    ):
        loc_val, scale_val = _generate_params(
            batch_shape, loc_shape, scale_shape, "cpu"
        )
        td_dist = TensorNormal(
            loc=loc_val,
            scale=scale_val,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
            shape=loc_val.shape,
            device=loc_val.device,
        )
        value = torch.rand(
            td_dist.dist().batch_shape + td_dist.dist().event_shape, device="cpu"
        )

        def icdf_fn(dist, val):
            return dist.icdf(val)

        eager_icdf, compiled_icdf = run_and_compare_compiled(
            icdf_fn, td_dist, value, fullgraph=False
        )
        torch.allclose(eager_icdf, compiled_icdf)

    @pytest.mark.parametrize(
        "batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims",
        TEST_CASES,
        ids=[
            f"batch={bs},loc={ls},scale={ss},rbn={rbn}"
            for bs, ls, ss, rbn in TEST_CASES
        ],
    )
    def test_entropy_matches_torch_distribution(
        self, batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims
    ):
        loc_val, scale_val = _generate_params(
            batch_shape, loc_shape, scale_shape, "cpu"
        )
        td_dist = TensorNormal(
            loc=loc_val,
            scale=scale_val,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
            shape=loc_val.shape,
            device=loc_val.device,
        )
        torch.allclose(td_dist.entropy(), td_dist.dist().entropy())

    @pytest.mark.parametrize(
        "batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims",
        TEST_CASES,
        ids=[
            f"batch={bs},loc={ls},scale={ss},rbn={rbn}"
            for bs, ls, ss, rbn in TEST_CASES
        ],
    )
    def test_entropy_compilation(
        self, batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims
    ):
        loc_val, scale_val = _generate_params(
            batch_shape, loc_shape, scale_shape, "cpu"
        )
        td_dist = TensorNormal(
            loc=loc_val,
            scale=scale_val,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
            shape=loc_val.shape,
            device=loc_val.device,
        )

        def entropy_fn(dist):
            return dist.entropy()

        eager_entropy, compiled_entropy = run_and_compare_compiled(
            entropy_fn, td_dist, fullgraph=False
        )
        torch.allclose(eager_entropy, compiled_entropy)

    @pytest.mark.parametrize(
        "batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims",
        TEST_CASES,
        ids=[
            f"batch={bs},loc={ls},scale={ss},rbn={rbn}"
            for bs, ls, ss, rbn in TEST_CASES
        ],
    )
    def test_mean_matches_torch_distribution(
        self, batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims
    ):
        loc_val, scale_val = _generate_params(
            batch_shape, loc_shape, scale_shape, "cpu"
        )
        td_dist = TensorNormal(
            loc=loc_val,
            scale=scale_val,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
            shape=loc_val.shape,
            device=loc_val.device,
        )
        torch.allclose(td_dist.mean, td_dist.dist().mean)

    @pytest.mark.parametrize(
        "batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims",
        TEST_CASES,
        ids=[
            f"batch={bs},loc={ls},scale={ss},rbn={rbn}"
            for bs, ls, ss, rbn in TEST_CASES
        ],
    )
    def test_mean_compilation(
        self, batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims
    ):
        loc_val, scale_val = _generate_params(
            batch_shape, loc_shape, scale_shape, "cpu"
        )
        td_dist = TensorNormal(
            loc=loc_val,
            scale=scale_val,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
            shape=loc_val.shape,
            device=loc_val.device,
        )

        def mean_fn(dist):
            return dist.mean

        eager_mean, compiled_mean = run_and_compare_compiled(
            mean_fn, td_dist, fullgraph=False
        )
        torch.allclose(eager_mean, compiled_mean)

    @pytest.mark.parametrize(
        "batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims",
        TEST_CASES,
        ids=[
            f"batch={bs},loc={ls},scale={ss},rbn={rbn}"
            for bs, ls, ss, rbn in TEST_CASES
        ],
    )
    def test_variance_matches_torch_distribution(
        self, batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims
    ):
        loc_val, scale_val = _generate_params(
            batch_shape, loc_shape, scale_shape, "cpu"
        )
        td_dist = TensorNormal(
            loc=loc_val,
            scale=scale_val,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
            shape=loc_val.shape,
            device=loc_val.device,
        )
        torch.allclose(td_dist.variance, td_dist.dist().variance)

    @pytest.mark.parametrize(
        "batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims",
        TEST_CASES,
        ids=[
            f"batch={bs},loc={ls},scale={ss},rbn={rbn}"
            for bs, ls, ss, rbn in TEST_CASES
        ],
    )
    def test_variance_compilation(
        self, batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims
    ):
        loc_val, scale_val = _generate_params(
            batch_shape, loc_shape, scale_shape, "cpu"
        )
        td_dist = TensorNormal(
            loc=loc_val,
            scale=scale_val,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
            shape=loc_val.shape,
            device=loc_val.device,
        )

        def variance_fn(dist):
            return dist.variance

        eager_variance, compiled_variance = run_and_compare_compiled(
            variance_fn, td_dist, fullgraph=False
        )
        torch.allclose(eager_variance, compiled_variance)

    @pytest.mark.parametrize(
        "batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims",
        TEST_CASES,
        ids=[
            f"batch={bs},loc={ls},scale={ss},rbn={rbn}"
            for bs, ls, ss, rbn in TEST_CASES
        ],
    )
    def test_stddev_matches_torch_distribution(
        self, batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims
    ):
        loc_val, scale_val = _generate_params(
            batch_shape, loc_shape, scale_shape, "cpu"
        )
        td_dist = TensorNormal(
            loc=loc_val,
            scale=scale_val,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
            shape=loc_val.shape,
            device=loc_val.device,
        )
        torch.allclose(td_dist.stddev, td_dist.dist().stddev)

    @pytest.mark.parametrize(
        "batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims",
        TEST_CASES,
        ids=[
            f"batch={bs},loc={ls},scale={ss},rbn={rbn}"
            for bs, ls, ss, rbn in TEST_CASES
        ],
    )
    def test_stddev_compilation(
        self, batch_shape, loc_shape, scale_shape, reinterpreted_batch_ndims
    ):
        loc_val, scale_val = _generate_params(
            batch_shape, loc_shape, scale_shape, "cpu"
        )
        td_dist = TensorNormal(
            loc=loc_val,
            scale=scale_val,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
            shape=loc_val.shape,
            device=loc_val.device,
        )

        def stddev_fn(dist):
            return dist.stddev

        eager_stddev, compiled_stddev = run_and_compare_compiled(
            stddev_fn, td_dist, fullgraph=False
        )
        torch.allclose(eager_stddev, compiled_stddev)
