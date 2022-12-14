#!/usr/bin/env python3
from jaxtyping import Float
from torch import Tensor

from ._linear_operator import LinearOperator
from .sum_linear_operator import SumLinearOperator


class PsdSumLinearOperator(SumLinearOperator):
    """
    A SumLinearOperator, but where every component of the sum is positive semi-definite
    """

    def zero_mean_mvn_samples(
        self: Float[LinearOperator, "*batch N N"], num_samples: int
    ) -> Float[Tensor, "num_samples *batch N"]:
        return sum(linear_op.zero_mean_mvn_samples(num_samples) for linear_op in self.linear_ops)
