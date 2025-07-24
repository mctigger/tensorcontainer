from __future__ import annotations

from tensorcontainer.distributions.soft_bernoulli import SoftBernoulli
from tensorcontainer.tensor_distribution.bernoulli import TensorBernoulli


class TensorSoftBernoulli(TensorBernoulli):
    def dist(self):
        return SoftBernoulli(
            logits=self._logits, probs=self._probs, validate_args=self._validate_args
        )
