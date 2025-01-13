#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import BlockDiagonalSparseLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class TestBlockDiagonalSparseLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    def create_linear_op(self):
        size_input_dim = 10
        non_zero_idcs = torch.tensor([[0, 2, 4, 5], [3, 7, 2, 1]], dtype=torch.int64, requires_grad=False)
        blocks = torch.tensor([[1.0, 2, 6, -2], [9.4, -1.0, 0.0, 2.0]], dtype=torch.float, requires_grad=True)

        return BlockDiagonalSparseLinearOperator(
            non_zero_idcs=non_zero_idcs, blocks=blocks, size_input_dim=size_input_dim
        )

    def evaluate_linear_op(self, linear_op):
        return torch.zeros((linear_op.blocks.shape[0], linear_op.size_input_dim)).scatter_(
            src=linear_op.blocks, index=linear_op.non_zero_idcs, dim=1
        )


if __name__ == "__main__":
    unittest.main()
