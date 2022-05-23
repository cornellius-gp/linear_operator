#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import LowRankRootLinearOperator
from linear_operator.test.linear_operator_test_case import RectangularLinearOperatorTestCase


class TestLowRankRootLinearOperator(RectangularLinearOperatorTestCase, unittest.TestCase):
    def create_lazy_tensor(self):
        root = torch.randn(3, 1, requires_grad=True)
        return LowRankRootLinearOperator(root)

    def evaluate_lazy_tensor(self, lazy_tensor):
        root = lazy_tensor.root.tensor
        res = root.matmul(root.transpose(-1, -2))
        return res


class TestLowRankRootLinearOperatorBatch(TestLowRankRootLinearOperator):
    seed = 1

    def create_lazy_tensor(self):
        root = torch.randn(3, 5, 2)
        return LowRankRootLinearOperator(root)


class TestLowRankRootLinearOperatorMultiBatch(TestLowRankRootLinearOperator):
    seed = 1
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def create_lazy_tensor(self):
        root = torch.randn(4, 3, 5, 2)
        return LowRankRootLinearOperator(root)


if __name__ == "__main__":
    unittest.main()
