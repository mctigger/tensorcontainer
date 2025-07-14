from __future__ import annotations

from torch.distributions import Independent

from tensorcontainer.tensor_distribution.base import TensorDistribution


class TensorIndependent(TensorDistribution):
    base_distribution: TensorDistribution
    reinterpreted_batch_ndims: int

    def __init__(
        self, base_distribution: TensorDistribution, reinterpreted_batch_ndims: int
    ):
        self.base_distribution = base_distribution
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

        super().__init__(
            base_distribution.shape[:-reinterpreted_batch_ndims],
            base_distribution.device,
        )

    def dist(self):
        return Independent(
            self.base_distribution.dist(), self.reinterpreted_batch_ndims
        )
