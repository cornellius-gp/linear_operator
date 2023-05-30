#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import BlockTensorLinearOperator
from linear_operator.test.base_test_case import BaseTestCase

# from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class TestBlockBlockSimple(BaseTestCase, unittest.TestCase):
    def test_multiply(self):
        T = 2
        N = 4
        M = 3
        K = 5

        A = torch.randn(T, T, N, M)
        B = torch.randn(T, T, M, K)

        A_blo = BlockTensorLinearOperator.from_tensor(A, T)
        B_blo = BlockTensorLinearOperator.from_tensor(B, T)
        res_AB = A_blo._matmul(B_blo)
        res_dense_AB = res_AB.to_dense()

        A_dense = A.permute(0, 2, 1, 3).reshape(T * N, T * M)
        B_dense = B.permute(0, 2, 1, 3).reshape(T * M, T * K)
        expected = A_dense @ B_dense
        self.assertAllClose(res_dense_AB, expected)
        self.assertAllClose(A_dense, A_blo.to_dense())
        self.assertAllClose(B_dense, B_blo.to_dense())

        # Try to convert dense to block
        Ne = A_dense.size(0) // T
        Me = A_dense.size(1) // T
        A_blocks_est = A_dense.reshape(T, Ne, T, Me)
        A_blocks_est = A_blocks_est.permute(0, 2, 1, 3)
        self.assertAllClose(A, A_blocks_est)

        # Check Tensor multiplication
        # res_tensor_AB = A_blo._matmul(B_dense)
        # res_tensor_dense_AB = res_tensor_AB.to_dense()
        # self.assertAllClose(res_dense_AB, res_tensor_dense_AB)


rem = """

class TestBlockBlockLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = False
    T = 2
    N = M = 4  # Try a square for this set of tests
    # N = 4
    # M = 3

    A_dense = torch.eye(T * N)
    A_blocks = A_dense.reshape(T, N, T, M).permute(0, 2, 1, 3)

    # A = torch.randn(T, T, N, M)  # Need to make something +ve definite

    def create_linear_op(self):
        A_blo = BlockBLockLinearOperator.from_tensor(self.A_blocks)
        return A_blo

    def evaluate_linear_op(self, linear_op):
        D = linear_op.to_dense()
        return D
"""


if __name__ == "__main__":
    unittest.main()