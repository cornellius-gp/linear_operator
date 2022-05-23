#!/usr/bin/env python3

from .sum_linear_operator import SumLinearOperator


class PsdSumLinearOperator(SumLinearOperator):
    """
    A SumLinearOperator, but where every component of the sum is positive semi-definite
    """

    def zero_mean_mvn_samples(self, num_samples):
        return sum(lazy_tensor.zero_mean_mvn_samples(num_samples) for lazy_tensor in self.lazy_tensors)
