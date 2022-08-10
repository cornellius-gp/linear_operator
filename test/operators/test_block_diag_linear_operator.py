#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import BlockDiagLinearOperator, DenseLinearOperator, DiagLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class TestBlockDiagLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    def create_linear_op(self):
        blocks = torch.randn(8, 4, 4)
        blocks = blocks.matmul(blocks.mT)
        blocks.add_(torch.eye(4, 4).unsqueeze_(0))
        return BlockDiagLinearOperator(DenseLinearOperator(blocks))

    def evaluate_linear_op(self, linear_op):
        blocks = linear_op.base_linear_op.tensor
        actual = torch.zeros(32, 32)
        for i in range(8):
            actual[i * 4 : (i + 1) * 4, i * 4 : (i + 1) * 4] = blocks[i]
        return actual


class TestBlockDiagLinearOperatorBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    def create_linear_op(self):
        blocks = torch.randn(2, 6, 4, 4)
        blocks = blocks.matmul(blocks.mT)
        blocks.add_(torch.eye(4, 4))
        return BlockDiagLinearOperator(DenseLinearOperator(blocks), block_dim=2)

    def evaluate_linear_op(self, linear_op):
        blocks = linear_op.base_linear_op.tensor
        actual = torch.zeros(2, 24, 24)
        for i in range(2):
            for j in range(6):
                actual[i, j * 4 : (j + 1) * 4, j * 4 : (j + 1) * 4] = blocks[i, j]
        return actual


class TestBlockDiagLinearOperatorMultiBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def create_linear_op(self):
        blocks = torch.randn(2, 6, 5, 4, 4)
        blocks = blocks.matmul(blocks.mT)
        blocks.add_(torch.eye(4, 4))
        blocks.detach_()
        return BlockDiagLinearOperator(DenseLinearOperator(blocks), block_dim=1)

    def evaluate_linear_op(self, linear_op):
        blocks = linear_op.base_linear_op.tensor
        actual = torch.zeros(2, 5, 24, 24)
        for i in range(2):
            for j in range(6):
                for k in range(5):
                    actual[i, k, j * 4 : (j + 1) * 4, j * 4 : (j + 1) * 4] = blocks[i, k, j]
        return actual


class TestBlockDiagLinearOperatorMetaClass(unittest.TestCase):
    def test_metaclass_constructor(self):
        k, n = 3, 5  # number of blocks, block size
        b1, b2 = 2, 3  # batch dimensions
        base_operators = [torch.randn(k, n), torch.randn(b1, b2, k, n)]
        subtest_names = ["non-batched input", "batched input"]
        # repeats tests for both batched and non-batched tensors
        for (base_op, test_name) in zip(base_operators, subtest_names):
            with self.subTest(test_name):
                base_diag = DiagLinearOperator(base_op)
                linear_op = BlockDiagLinearOperator(base_diag)

                # checks that metaclass constructor returns a diagonal operator
                # if a DiagLinearOperator is passed to BlockDiagLinearOperator
                self.assertEqual(type(linear_op), DiagLinearOperator)

                # matrix-vector-multiplication test
                diag_values = base_op.flatten(start_dim=-2)  # operator of non-block diagonal values
                x = torch.randn_like(diag_values)
                self.assertTrue(torch.equal(diag_values * x, (linear_op @ x.unsqueeze(-1)).squeeze(-1)))

                # checks that the representation is numerically accurate
                dense_operator = linear_op.to_dense()
                truth_operator = torch.diag_embed(diag_values)  # creates batch of diagonal operators
                self.assertTrue(torch.equal(dense_operator, truth_operator))

                with self.assertRaisesRegex(NotImplementedError, "with block_dim = -2 != -3 is not supported"):
                    # beside the dimensions not working out here, this should never
                    # be allowed as long as there is no special case for it, because
                    # matmuls with the resulting object will fail
                    BlockDiagLinearOperator(base_diag, block_dim=-2)


if __name__ == "__main__":
    unittest.main()
