#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import MatmulLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase, RectangularLinearOperatorTestCase


class TestMatmulLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 1

    def create_linear_op(self):
        lhs = torch.randn(5, 6, requires_grad=True)
        rhs = lhs.clone().detach().mT
        covar = MatmulLinearOperator(lhs, rhs)
        return covar

    def evaluate_linear_op(self, linear_op):
        return linear_op.left_linear_op.tensor.matmul(linear_op.right_linear_op.tensor)


class TestMatmulLinearOperatorBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 3

    def create_linear_op(self):
        lhs = torch.randn(5, 5, 6, requires_grad=True)
        rhs = lhs.clone().detach().mT
        covar = MatmulLinearOperator(lhs, rhs)
        return covar

    def evaluate_linear_op(self, linear_op):
        return linear_op.left_linear_op.tensor.matmul(linear_op.right_linear_op.tensor)


class TestMatmulLinearOperatorRectangular(RectangularLinearOperatorTestCase, unittest.TestCase):
    def create_linear_op(self):
        lhs = torch.randn(5, 3, requires_grad=True)
        rhs = torch.randn(3, 6, requires_grad=True)
        covar = MatmulLinearOperator(lhs, rhs)
        return covar

    def evaluate_linear_op(self, linear_op):
        return linear_op.left_linear_op.tensor.matmul(linear_op.right_linear_op.tensor)


class TestMatmulLinearOperatorRectangularMultiBatch(RectangularLinearOperatorTestCase, unittest.TestCase):
    def create_linear_op(self):
        lhs = torch.randn(2, 3, 5, 3, requires_grad=True)
        rhs = torch.randn(2, 3, 3, 6, requires_grad=True)
        covar = MatmulLinearOperator(lhs, rhs)
        return covar

    def evaluate_linear_op(self, linear_op):
        return linear_op.left_linear_op.tensor.matmul(linear_op.right_linear_op.tensor)


if __name__ == "__main__":
    unittest.main()
