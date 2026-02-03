#!/usr/bin/env python3
from torch import Tensor

from linear_operator.operators._linear_operator import LinearOperator
from linear_operator.operators.sum_linear_operator import SumLinearOperator


class PsdSumLinearOperator(SumLinearOperator):
    """
    A SumLinearOperator, but where every component of the sum is positive semi-definite
    """

    def zero_mean_mvn_samples(
        self: LinearOperator, num_samples: int  # shape: (*batch, N, N)
    ) -> Tensor:  # shape: (num_samples, *batch, N)
        return sum(linear_op.zero_mean_mvn_samples(num_samples) for linear_op in self.linear_ops)
