#!/usr/bin/env python3

import unittest

import torch

import linear_operator
from linear_operator.operators import DenseLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class TestDenseLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_op(self):
        mat = torch.randn(5, 6)
        mat = mat.matmul(mat.mT)
        mat.requires_grad_(True)
        return DenseLinearOperator(mat)

    def evaluate_linear_op(self, linear_op):
        return linear_op.tensor

    def test_root_decomposition_exact(self):
        linear_op = self.create_linear_op()
        test_mat = torch.randn(*linear_op.batch_shape, linear_op.size(-1), 5)
        with linear_operator.settings.fast_computations(covar_root_decomposition=False):
            root_approx = linear_op.root_decomposition()
            res = root_approx.matmul(test_mat)
            actual = linear_op.matmul(test_mat)
            self.assertLess(torch.norm(res - actual) / actual.norm(), 0.1)


class TestDenseLinearOperatorBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_op(self):
        mat = torch.randn(3, 5, 6)
        mat = mat.matmul(mat.mT)
        mat.requires_grad_(True)
        return DenseLinearOperator(mat)

    def evaluate_linear_op(self, linear_op):
        return linear_op.tensor

    def test_root_decomposition_exact(self):
        linear_op = self.create_linear_op()
        test_mat = torch.randn(*linear_op.batch_shape, linear_op.size(-1), 5)
        with linear_operator.settings.fast_computations(covar_root_decomposition=False):
            root_approx = linear_op.root_decomposition()
            res = root_approx.matmul(test_mat)
            actual = linear_op.matmul(test_mat)
            self.assertLess(torch.norm(res - actual) / actual.norm(), 0.1)


class TestDenseLinearOperatorMultiBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def create_linear_op(self):
        mat = torch.randn(2, 3, 5, 6)
        mat = mat.matmul(mat.mT)
        mat.requires_grad_(True)
        return DenseLinearOperator(mat)

    def evaluate_linear_op(self, linear_op):
        return linear_op.tensor
