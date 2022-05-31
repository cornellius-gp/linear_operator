#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import RootLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


def make_random_mat(size, rank, batch_shape=torch.Size(())):
    res = torch.randn(*batch_shape, size, rank)
    return res


class TestMulLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 10

    def create_linear_op(self):
        mat1 = make_random_mat(6, 6)
        mat2 = make_random_mat(6, 6)
        res = torch.mul(RootLinearOperator(mat1), RootLinearOperator(mat2))
        return res.add_diagonal(torch.tensor(2.0))

    def evaluate_linear_op(self, linear_op):
        diag_tensor = linear_op._diag_tensor.to_dense()
        res = torch.mul(linear_op._linear_op.left_linear_op.to_dense(), linear_op._linear_op.right_linear_op.to_dense())
        res = res + diag_tensor
        return res


class TestMulLinearOperatorBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 2

    def create_linear_op(self):
        mat1 = make_random_mat(6, rank=6, batch_shape=torch.Size((2,)))
        mat2 = make_random_mat(6, rank=6, batch_shape=torch.Size((2,)))
        res = RootLinearOperator(mat1) * RootLinearOperator(mat2)
        return res.add_diagonal(torch.tensor(2.0))

    def evaluate_linear_op(self, linear_op):
        diag_tensor = linear_op._diag_tensor.to_dense()
        res = torch.mul(linear_op._linear_op.left_linear_op.to_dense(), linear_op._linear_op.right_linear_op.to_dense())
        res = res + diag_tensor
        return res


class TestMulLinearOperatorMultiBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 1
    skip_slq_tests = True

    def create_linear_op(self):
        mat1 = make_random_mat(6, rank=6, batch_shape=torch.Size((2, 3)))
        mat2 = make_random_mat(6, rank=6, batch_shape=torch.Size((2, 3)))
        res = RootLinearOperator(mat1) * RootLinearOperator(mat2)
        return res.add_diagonal(torch.tensor(0.5))

    def evaluate_linear_op(self, linear_op):
        diag_tensor = linear_op._diag_tensor.to_dense()
        res = torch.mul(linear_op._linear_op.left_linear_op.to_dense(), linear_op._linear_op.right_linear_op.to_dense())
        res = res + diag_tensor
        return res

    def test_inv_quad_logdet(self):
        pass


if __name__ == "__main__":
    unittest.main()
