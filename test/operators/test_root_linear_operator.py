#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import RootLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class TestRootLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True
    should_call_lanczos = False
    should_call_lanczos_diagonalization = True

    def create_linear_op(self):
        root = torch.randn(3, 5, requires_grad=True)
        return RootLinearOperator(root)

    def evaluate_linear_op(self, linear_op):
        root = linear_op.root.tensor
        res = root.matmul(root.mT)
        return res


class TestRootLinearOperatorBatch(TestRootLinearOperator):
    seed = 1

    def create_linear_op(self):
        root = torch.randn(3, 5, 5) + torch.eye(5)
        root.requires_grad_(True)
        return RootLinearOperator(root)


class TestRootLinearOperatorMultiBatch(TestRootLinearOperator):
    seed = 2
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def create_linear_op(self):
        root = torch.randn(2, 3, 5, 5) + torch.eye(5)
        root.requires_grad_(True)
        return RootLinearOperator(root)


if __name__ == "__main__":
    unittest.main()
