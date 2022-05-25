#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import DenseLinearOperator, PsdSumLinearOperator, ToeplitzLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class TestPsdSumLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    def create_linear_op(self):
        c1 = torch.tensor([5, 1, 2, 0], dtype=torch.float, requires_grad=True)
        t1 = ToeplitzLinearOperator(c1)
        c2 = torch.tensor([6, 0, 1, -1], dtype=torch.float, requires_grad=True)
        t2 = ToeplitzLinearOperator(c2)
        return PsdSumLinearOperator(t1, t2)

    def evaluate_linear_op(self, linear_op):
        tensors = [lt.to_dense() for lt in linear_op.linear_ops]
        return sum(tensors)


class TestPsdSumLinearOperatorBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    def create_linear_op(self):
        c1 = torch.tensor([[2, 0.5, 0, 0], [5, 1, 2, 0]], dtype=torch.float, requires_grad=True)
        t1 = ToeplitzLinearOperator(c1)
        c2 = torch.tensor([[2, 0.5, 0, 0], [6, 0, 1, -1]], dtype=torch.float, requires_grad=True)
        t2 = ToeplitzLinearOperator(c2)
        return PsdSumLinearOperator(t1, t2)

    def evaluate_linear_op(self, linear_op):
        tensors = [lt.to_dense() for lt in linear_op.linear_ops]
        return sum(tensors)


class TestPsdSumLinearOperatorMultiBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def create_linear_op(self):
        mat1 = torch.randn(2, 3, 4, 4)
        lt1 = DenseLinearOperator(mat1 @ mat1.mT)
        mat2 = torch.randn(2, 3, 4, 4)
        lt2 = DenseLinearOperator(mat2 @ mat2.mT)
        return PsdSumLinearOperator(lt1, lt2)

    def evaluate_linear_op(self, linear_op):
        tensors = [lt.to_dense() for lt in linear_op.linear_ops]
        return sum(tensors)


if __name__ == "__main__":
    unittest.main()
