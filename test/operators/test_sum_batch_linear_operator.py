#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import DenseLinearOperator, SumBatchLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class TestSumBatchLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 6
    should_test_sample = True

    def create_linear_op(self):
        blocks = torch.randn(12, 4, 4)
        blocks = blocks.mT.matmul(blocks)
        blocks.requires_grad_(True)
        return SumBatchLinearOperator(DenseLinearOperator(blocks))

    def evaluate_linear_op(self, linear_op):
        blocks = linear_op.base_linear_op.tensor
        return blocks.sum(0)


class TestSumBatchLinearOperatorBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 6
    should_test_sample = True

    def create_linear_op(self):
        blocks = torch.randn(2, 6, 4, 4)
        blocks = blocks.mT.matmul(blocks)
        blocks.requires_grad_(True)
        return SumBatchLinearOperator(DenseLinearOperator(blocks))

    def evaluate_linear_op(self, linear_op):
        blocks = linear_op.base_linear_op.tensor
        return blocks.view(2, 6, 4, 4).sum(1)


class TestSumBatchLinearOperatorMultiBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 6
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def create_linear_op(self):
        blocks = torch.randn(2, 3, 6, 4, 4)
        blocks = blocks.mT.matmul(blocks)
        blocks.detach_()
        return SumBatchLinearOperator(DenseLinearOperator(blocks), block_dim=1)

    def evaluate_linear_op(self, linear_op):
        blocks = linear_op.base_linear_op.tensor
        return blocks.sum(-3)


if __name__ == "__main__":
    unittest.main()
