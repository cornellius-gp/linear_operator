#!/usr/bin/env python3

import itertools
import unittest

import torch

from linear_operator.operators import DenseLinearOperator, MulLinearOperator, RootLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


def make_random_mat(size, rank, batch_shape=torch.Size(())):
    res = torch.randn(*batch_shape, size, rank)
    return res


class BatchConstantMulMixin:
    def test_constant_mul_batch(self):
        batch_shape = self.create_linear_op().batch_shape
        mats = [torch.randn(*batch_shape, 6, 6) for _ in range(2)]
        mul_linear_ops = [
            # Root decompositions given as RootLinearOperators.
            self.create_linear_op()._linear_op,
            # Cholesky (triangular) root decompositions of dense matrices.
            MulLinearOperator(*(DenseLinearOperator(mat @ mat.mT + torch.eye(6)) for mat in mats)),
        ]
        positive = torch.rand(*batch_shape, 1, 1) + 0.5
        mixed = positive.clone()
        mixed.view(-1)[0] *= -1
        for mul_linear_op, constant in itertools.product(mul_linear_ops, (positive, mixed)):
            res = mul_linear_op * constant
            # Positive constants keep the root decompositions.
            self.assertEqual(isinstance(res, MulLinearOperator), constant is positive)
            self.assertAllClose(res.to_dense(), mul_linear_op.to_dense() * constant)


class TestMulLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 10

    def create_linear_op(self):
        mat1 = make_random_mat(6, 6)
        mat2 = make_random_mat(6, 6)
        res = torch.mul(RootLinearOperator(mat1), RootLinearOperator(mat2))
        return res.add_diagonal(torch.tensor(2.0))

    def evaluate_linear_op(self, linear_op):
        diag_tensor = linear_op._diag_tensor.to_dense()
        res = torch.mul(
            linear_op._linear_op.left_linear_op.to_dense(),
            linear_op._linear_op.right_linear_op.to_dense(),
        )
        res = res + diag_tensor
        return res


class TestMulLinearOperatorBatch(BatchConstantMulMixin, LinearOperatorTestCase, unittest.TestCase):
    seed = 2

    def create_linear_op(self):
        mat1 = make_random_mat(6, rank=6, batch_shape=torch.Size((2,)))
        mat2 = make_random_mat(6, rank=6, batch_shape=torch.Size((2,)))
        res = RootLinearOperator(mat1) * RootLinearOperator(mat2)
        return res.add_diagonal(torch.tensor(2.0))

    def evaluate_linear_op(self, linear_op):
        diag_tensor = linear_op._diag_tensor.to_dense()
        res = torch.mul(
            linear_op._linear_op.left_linear_op.to_dense(),
            linear_op._linear_op.right_linear_op.to_dense(),
        )
        res = res + diag_tensor
        return res


class TestMulLinearOperatorMultiBatch(BatchConstantMulMixin, LinearOperatorTestCase, unittest.TestCase):
    seed = 1
    skip_slq_tests = True

    def create_linear_op(self):
        mat1 = make_random_mat(6, rank=6, batch_shape=torch.Size((2, 3)))
        mat2 = make_random_mat(6, rank=6, batch_shape=torch.Size((2, 3)))
        res = RootLinearOperator(mat1) * RootLinearOperator(mat2)
        return res.add_diagonal(torch.tensor(0.5))

    def evaluate_linear_op(self, linear_op):
        diag_tensor = linear_op._diag_tensor.to_dense()
        res = torch.mul(
            linear_op._linear_op.left_linear_op.to_dense(),
            linear_op._linear_op.right_linear_op.to_dense(),
        )
        res = res + diag_tensor
        return res

    def test_inv_quad_logdet(self):
        pass


if __name__ == "__main__":
    unittest.main()
