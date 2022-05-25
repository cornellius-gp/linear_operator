#!/usr/bin/env python3

import unittest

import torch

from linear_operator import to_linear_operator
from linear_operator.operators import BatchRepeatLinearOperator, ToeplitzLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase, RectangularLinearOperatorTestCase


class TestBatchRepeatLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_op(self):
        toeplitz_column = torch.tensor([4, 0.1, 0.05, 0.01, 0.0], dtype=torch.float)
        toeplitz_column.detach_()
        return BatchRepeatLinearOperator(ToeplitzLinearOperator(toeplitz_column), torch.Size((3,)))

    def evaluate_linear_op(self, linear_op):
        evaluated = linear_op.base_linear_op.to_dense()
        return evaluated.repeat(*linear_op.batch_repeat, 1, 1)


class TestBatchRepeatLinearOperatorNonSquare(RectangularLinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_op(self):
        rand_mat = torch.randn(25, 12, dtype=torch.float)
        rand_mat.detach_()
        return BatchRepeatLinearOperator(to_linear_operator(rand_mat), torch.Size((10,)))

    def evaluate_linear_op(self, linear_op):
        evaluated = linear_op.base_linear_op.to_dense()
        return evaluated.repeat(*linear_op.batch_repeat, 1, 1)


class TestBatchRepeatLinearOperatorBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_op(self):
        toeplitz_column = torch.tensor([[4, 0, 0, 1], [3, 0, -0.5, -1]], dtype=torch.float)
        toeplitz_column.detach_()
        return BatchRepeatLinearOperator(ToeplitzLinearOperator(toeplitz_column), torch.Size((3,)))
        return BatchRepeatLinearOperator(ToeplitzLinearOperator(toeplitz_column), torch.Size((3,)))

    def evaluate_linear_op(self, linear_op):
        evaluated = linear_op.base_linear_op.to_dense()
        return evaluated.repeat(*linear_op.batch_repeat, 1, 1)


class TestBatchRepeatLinearOperatorMultiBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    skip_slq_tests = True

    def create_linear_op(self):
        toeplitz_column = torch.tensor(
            [[[4, 0, 0, 1], [3, 0, -0.5, -1]], [[2, 0.1, 0.01, 0.0], [3, 0, -0.1, -2]]], dtype=torch.float
        )
        toeplitz_column.detach_()
        return BatchRepeatLinearOperator(ToeplitzLinearOperator(toeplitz_column), torch.Size((2, 3, 1, 4)))

    def evaluate_linear_op(self, linear_op):
        evaluated = linear_op.base_linear_op.to_dense()
        return evaluated.repeat(*linear_op.batch_repeat, 1, 1)


if __name__ == "__main__":
    unittest.main()
