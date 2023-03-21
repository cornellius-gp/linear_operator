#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import LowRankRootLinearOperator
from linear_operator.test.linear_operator_test_case import RectangularLinearOperatorTestCase


class TestLowRankRootLinearOperator(RectangularLinearOperatorTestCase, unittest.TestCase):
    def create_linear_op(self):
        root = torch.randn(3, 1, requires_grad=True)
        return LowRankRootLinearOperator(root)

    def evaluate_linear_op(self, linear_op):
        root = linear_op.root.tensor
        res = root.matmul(root.mT)
        return res


class TestLowRankRootLinearOperatorBatch(TestLowRankRootLinearOperator):
    seed = 1

    def create_linear_op(self):
        root = torch.randn(3, 5, 2)
        return LowRankRootLinearOperator(root)


class TestLowRankRootLinearOperatorMultiBatch(TestLowRankRootLinearOperator):
    seed = 1
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def create_linear_op(self):
        root = torch.randn(4, 3, 5, 2)
        return LowRankRootLinearOperator(root)


if __name__ == "__main__":
    unittest.main()
