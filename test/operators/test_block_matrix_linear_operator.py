#!/usr/bin/env python3
import itertools
import unittest

import torch

from linear_operator.operators import BlockMatrixLinearOperator
from linear_operator.test.base_test_case import BaseTestCase
from linear_operator.test.linear_operator_core_test_case import CoreLinearOperatorTestCase


class TestBlockTensorSimple(BaseTestCase, unittest.TestCase):
    def dense_to_4d(self, A_dense, T):
        Ne = A_dense.size(0) // T
        Me = A_dense.size(1) // T
        A_blocks_est = A_dense.reshape(T, Ne, T, Me)
        A_blocks_est = A_blocks_est.permute(0, 2, 1, 3)
        return A_blocks_est

    def test_multiply(self):
        T = 2
        N = 4
        M = 3
        K = 5

        A = torch.randn(T, T, N, M)
        B = torch.randn(T, T, M, K)

        A_blo = BlockMatrixLinearOperator.from_tensor(A, T)
        B_blo = BlockMatrixLinearOperator.from_tensor(B, T)
        res_AB = A_blo.matmul(B_blo)
        res_dense_AB = res_AB.to_dense()

        A_dense = A.permute(0, 2, 1, 3).reshape(T * N, T * M)
        B_dense = B.permute(0, 2, 1, 3).reshape(T * M, T * K)
        expected = A_dense @ B_dense
        self.assertAllClose(res_dense_AB, expected)
        self.assertAllClose(A_dense, A_blo.to_dense())
        self.assertAllClose(B_dense, B_blo.to_dense())

        # Convert dense format back to blocks and compare
        A_blocks_est = self.dense_to_4d(A_dense, T)
        self.assertAllClose(A, A_blocks_est)

        # Check Tensor multiplication
        res_tensor_AB = A_blo.matmul(B_dense)
        res_tensor_dense_AB = res_tensor_AB.to_dense()
        self.assertAllClose(res_dense_AB, res_tensor_dense_AB)

    def test_sparse_multiply(self):
        T, N, M = 2, 4, 3
        As = [torch.rand(N, M) for _ in range(T)]
        Bs = [[torch.rand(M, M) for _ in range(T)] for _ in range(T)]
        Cs = [torch.rand(N, N) for _ in range(T)]
        # L = torch.rand(T, T)

        A_dense = torch.zeros((N * T, M * T))  # BlockDiag (non-square)
        B_dense = torch.zeros((M * T, M * T))  # Dense
        C_dense = torch.zeros((N * T, N * T))  # BlockDiag
        # L_dense = torch.kron(L, torch.eye(N))  # Kroneker

        for t in range(T):
            A_dense[N * t : N * (t + 1), M * t : M * (t + 1)] = As[t]
            C_dense[N * t : N * (t + 1), N * t : N * (t + 1)] = Cs[t]

        for t1, t2 in itertools.product(range(T), range(T)):
            B_dense[M * t1 : M * (t1 + 1), M * t2 : M * (t2 + 1)] = Bs[t1][t2]

        # Convert dense formats to blocks
        A = self.dense_to_4d(A_dense, T)
        B = self.dense_to_4d(B_dense, T)

        # A_blo will contain dense operators along the diagonal + Zero operators off diagonal
        A_blo = BlockMatrixLinearOperator.from_tensor(A, T)
        B_blo = BlockMatrixLinearOperator.from_tensor(B, T)
        res_AB = A_blo.matmul(B_blo)
        res_dense_AB = res_AB.to_dense()

        expected = A_dense @ B_dense
        self.assertAllClose(res_dense_AB, expected)
        self.assertAllClose(A_dense, A_blo.to_dense())
        self.assertAllClose(B_dense, B_blo.to_dense())


class TestLinearOperatorBlockTensorLinearOperator(CoreLinearOperatorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = False
    T = 2
    N = M = 4  # Try a square for this set of tests

    A_dense = torch.eye(T * N)
    A_blocks = A_dense.reshape(T, N, T, M).permute(0, 2, 1, 3)

    def create_linear_op(self):
        A_blo = BlockMatrixLinearOperator.from_tensor(self.A_blocks, self.T)
        return A_blo

    def evaluate_linear_op(self, linear_op):
        D = linear_op.to_dense()
        return D


if __name__ == "__main__":
    unittest.main()
