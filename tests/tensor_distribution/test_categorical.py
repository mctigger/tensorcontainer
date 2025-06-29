import pytest
import torch
import torch.testing
from torch.distributions import Independent, OneHotCategorical

from tensorcontainer.tensor_distribution import TensorCategorical
from tests.compile_utils import run_and_compare_compiled


TEST_CASES = [
    ((), (5,), 1),
    ((3,), (5,), 1),
    ((2, 4), (5,), 1),
    ((2, 3), (4, 5), 2),
]


class TestTensorCategorical:
    def _run_and_verify(self, td_categorical, batch_shape, event_shape):
        """
        Central helper method to encapsulate common testing logic for TensorCategorical.
        """
        # 1. Verify .dist() property
        torch_dist = td_categorical.dist()
        assert isinstance(torch_dist, Independent)
        assert isinstance(torch_dist.base_dist, OneHotCategorical)
        assert torch_dist.batch_shape == batch_shape
        assert torch_dist.event_shape == event_shape

        # 2. Test .sample()
        def _sample(td_cat):
            return td_cat.sample()

        sample, _ = run_and_compare_compiled(_sample, td_categorical, fullgraph=False)
        assert sample.shape == batch_shape + event_shape

        # 3. Test .rsample()
        def _rsample(td_cat):
            return td_cat.rsample()

        rsample, _ = run_and_compare_compiled(_rsample, td_categorical, fullgraph=False)
        assert rsample.shape == batch_shape + event_shape
        assert rsample.requires_grad == td_categorical.logits.requires_grad

        # 4. Test .log_prob()
        def _log_prob(td_cat, s):
            return td_cat.log_prob(s)

        log_prob, _ = run_and_compare_compiled(
            _log_prob, td_categorical, sample, fullgraph=False
        )
        assert log_prob.shape == batch_shape

    @pytest.mark.parametrize(
        "batch_shape, event_shape, reinterpreted_batch_ndims",
        TEST_CASES,
        ids=[
            f"batch={bs},event={es},reinterpreted={rbn}" for bs, es, rbn in TEST_CASES
        ],
    )
    def test_categorical_ops(self, batch_shape, event_shape, reinterpreted_batch_ndims):
        """Tests core operations (sample, rsample, log_prob) for TensorCategorical.

        This test verifies the correctness and torch.compile compatibility of
        the .sample(), .rsample(), and .log_prob() methods of TensorCategorical.

        Example with torch.Tensor:
            >>> logits = torch.randn(2, 3, 5) # batch_shape=(2,3), event_shape=(5,)
            >>> td_cat = TensorCategorical(logits=logits, reinterpreted_batch_ndims=1)
            >>> sample = td_cat.sample()
            >>> assert sample.shape == (2, 3, 5)
            >>> log_prob = td_cat.log_prob(sample)
            >>> assert log_prob.shape == (2, 3)
        """
        num_classes = event_shape[-1] if len(event_shape) > 0 else 1
        logits_shape = batch_shape + event_shape
        if len(event_shape) == 0:  # Scalar event shape case
            logits_shape = batch_shape + (num_classes,)

        logits = torch.randn(*logits_shape, requires_grad=True)

        td_categorical = TensorCategorical(
            logits=logits,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
            shape=logits.shape,
            device=logits.device,
        )
        self._run_and_verify(td_categorical, batch_shape, event_shape)
