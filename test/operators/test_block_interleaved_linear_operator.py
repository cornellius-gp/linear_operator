#!/usr/bin/env python3

import unittest
from itertools import product

import torch

from linear_operator.operators import BlockInterleavedLinearOperator, DenseLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class _BlockSliceTestMixin:
    def test_getitem_slices(self):
        linear_op = self.create_linear_op()
        evaluated = self.evaluate_linear_op(linear_op)
        num_blocks, block_size = linear_op.num_blocks, linear_op.base_linear_op.size(-1)
        size = num_blocks * block_size
        slices = [
            slice(0, num_blocks),  # bounds are multiples of num_blocks
            slice(num_blocks, 3 * num_blocks),
            slice(num_blocks, size),
            slice(0, block_size),  # bounds are multiples of the block size
            slice(block_size, 2 * block_size),
            slice(1, num_blocks + 1),
            slice(-num_blocks, None),
            slice(None, num_blocks),
            slice(0, num_blocks, 1),
            slice(0, 2 * num_blocks, 2),
        ]
        for row_index, col_index in product(slices, repeat=2):
            with self.subTest(row_index=row_index, col_index=col_index):
                res = linear_op[..., row_index, col_index]
                self.assertAllClose(res.to_dense(), evaluated[..., row_index, col_index])
                if linear_op.dim() > 2:
                    res = linear_op[1, ..., row_index, col_index]
                    self.assertAllClose(res.to_dense(), evaluated[1, ..., row_index, col_index])

        # Slices whose bounds are multiples of num_blocks keep the interleaved structure
        res = linear_op[..., num_blocks : 3 * num_blocks, 0 : 2 * num_blocks]
        self.assertIsInstance(res, BlockInterleavedLinearOperator)
        self.assertEqual(res.num_blocks, num_blocks)


class TestBlockInterleavedLinearOperator(_BlockSliceTestMixin, LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    def create_linear_op(self):
        blocks = torch.randn(8, 4, 4)
        blocks = blocks.matmul(blocks.mT)
        blocks.add_(torch.eye(4, 4).unsqueeze_(0))
        return BlockInterleavedLinearOperator(DenseLinearOperator(blocks))

    def evaluate_linear_op(self, linear_op):
        blocks = linear_op.base_linear_op.tensor
        actual = torch.zeros(32, 32)
        for i in range(8):
            for j in range(4):
                for k in range(4):
                    actual[j * 8 + i, k * 8 + i] = blocks[i, j, k]
        return actual


class TestBlockInterleavedLinearOperatorBatch(_BlockSliceTestMixin, LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    def create_linear_op(self):
        blocks = torch.randn(2, 6, 4, 4)
        blocks = blocks.matmul(blocks.mT)
        blocks.add_(torch.eye(4, 4))
        return BlockInterleavedLinearOperator(DenseLinearOperator(blocks), block_dim=2)

    def evaluate_linear_op(self, linear_op):
        blocks = linear_op.base_linear_op.tensor
        actual = torch.zeros(2, 24, 24)
        for i in range(2):
            for j in range(6):
                for k in range(4):
                    for l in range(4):
                        actual[i, k * 6 + j, l * 6 + j] = blocks[i, j, k, l]
        return actual


class TestBlockInterleavedLinearOperatorMultiBatch(_BlockSliceTestMixin, LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def create_linear_op(self):
        blocks = torch.randn(2, 6, 5, 4, 4)
        blocks = blocks.matmul(blocks.mT)
        blocks.add_(torch.eye(4, 4))
        blocks.detach_()
        return BlockInterleavedLinearOperator(DenseLinearOperator(blocks), block_dim=1)

    def evaluate_linear_op(self, linear_op):
        blocks = linear_op.base_linear_op.tensor
        actual = torch.zeros(2, 5, 24, 24)
        for i in range(2):
            for j in range(6):
                for k in range(5):
                    for l in range(4):
                        for m in range(4):
                            actual[i, k, l * 6 + j, m * 6 + j] = blocks[i, k, j, l, m]
        return actual


if __name__ == "__main__":
    unittest.main()
