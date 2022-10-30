#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import CatLinearOperator, DenseLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class TestCatLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 1

    def create_linear_op(self):
        root = torch.randn(6, 7)
        self.psd_mat = root.matmul(root.t())

        slice1_mat = self.psd_mat[:2, :].requires_grad_()
        slice2_mat = self.psd_mat[2:4, :].requires_grad_()
        slice3_mat = self.psd_mat[4:6, :].requires_grad_()

        slice1 = DenseLinearOperator(slice1_mat)
        slice2 = DenseLinearOperator(slice2_mat)
        slice3 = DenseLinearOperator(slice3_mat)

        return CatLinearOperator(slice1, slice2, slice3, dim=-2)

    def evaluate_linear_op(self, linear_op):
        return self.psd_mat.detach().clone().requires_grad_()


class TestCatLinearOperatorColumn(LinearOperatorTestCase, unittest.TestCase):
    seed = 1

    def create_linear_op(self):
        root = torch.randn(6, 7)
        self.psd_mat = root.matmul(root.t())

        slice1_mat = self.psd_mat[:, :2].requires_grad_()
        slice2_mat = self.psd_mat[:, 2:4].requires_grad_()
        slice3_mat = self.psd_mat[:, 4:6].requires_grad_()

        slice1 = DenseLinearOperator(slice1_mat)
        slice2 = DenseLinearOperator(slice2_mat)
        slice3 = DenseLinearOperator(slice3_mat)

        return CatLinearOperator(slice1, slice2, slice3, dim=-1)

    def evaluate_linear_op(self, linear_op):
        return self.psd_mat.detach().clone().requires_grad_()


class TestCatLinearOperatorBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_op(self):
        root = torch.randn(3, 6, 7)
        self.psd_mat = root.matmul(root.mT)

        slice1_mat = self.psd_mat[..., :2, :].requires_grad_()
        slice2_mat = self.psd_mat[..., 2:4, :].requires_grad_()
        slice3_mat = self.psd_mat[..., 4:6, :].requires_grad_()

        slice1 = DenseLinearOperator(slice1_mat)
        slice2 = DenseLinearOperator(slice2_mat)
        slice3 = DenseLinearOperator(slice3_mat)

        return CatLinearOperator(slice1, slice2, slice3, dim=-2)

    def evaluate_linear_op(self, linear_op):
        return self.psd_mat.detach().clone().requires_grad_()


class TestCatLinearOperatorMultiBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    skip_slq_tests = True

    def create_linear_op(self):
        root = torch.randn(4, 3, 6, 7)
        self.psd_mat = root.matmul(root.mT)

        slice1_mat = self.psd_mat[..., :2, :].requires_grad_()
        slice2_mat = self.psd_mat[..., 2:4, :].requires_grad_()
        slice3_mat = self.psd_mat[..., 4:6, :].requires_grad_()

        slice1 = DenseLinearOperator(slice1_mat)
        slice2 = DenseLinearOperator(slice2_mat)
        slice3 = DenseLinearOperator(slice3_mat)

        return CatLinearOperator(slice1, slice2, slice3, dim=-2)

    def evaluate_linear_op(self, linear_op):
        return self.psd_mat.detach().clone().requires_grad_()


class TestCatLinearOperatorBatchCat(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    skip_slq_tests = True

    def create_linear_op(self):
        root = torch.randn(5, 3, 6, 7)
        self.psd_mat = root.matmul(root.mT)

        slice1_mat = self.psd_mat[:2, ...].requires_grad_()
        slice2_mat = self.psd_mat[2:3, ...].requires_grad_()
        slice3_mat = self.psd_mat[3:, ...].requires_grad_()

        slice1 = DenseLinearOperator(slice1_mat)
        slice2 = DenseLinearOperator(slice2_mat)
        slice3 = DenseLinearOperator(slice3_mat)

        return CatLinearOperator(slice1, slice2, slice3, dim=0)

    def evaluate_linear_op(self, linear_op):
        return self.psd_mat.detach().clone().requires_grad_()

    def test_getitem_broadcasted_tensor_index(self):
        linear_op = self.create_linear_op()

        with self.assertRaises(RuntimeError):
            linear_op[torch.tensor([0, 1, 1]).view(-1, 1), ...]


if __name__ == "__main__":
    unittest.main()
