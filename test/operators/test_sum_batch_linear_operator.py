#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import (
    DenseLinearOperator,
    DiagLinearOperator,
    KernelLinearOperator,
    LinearOperator,
    SumBatchLinearOperator,
)
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class _BroadcastInputLinearOperator(LinearOperator):
    def __init__(self, x1, x2, scale):
        super().__init__(x1, x2, scale)
        self.x1 = x1
        self.x2 = x2
        self.scale = scale

    def _matmul(self, rhs):
        mat = self.scale.unsqueeze(-1).unsqueeze(-1) * (self.x1 @ self.x2.mT)
        return mat @ rhs

    def _size(self):
        return torch.Size((*self.scale.shape, self.x1.size(-2), self.x2.size(-2)))

    def _transpose_nonbatch(self):
        return _BroadcastInputLinearOperator(self.x2, self.x1, self.scale)


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

    def test_sum_batch_broadcast_inputs_torch_cat_and_empty_slice(self):
        x = torch.randn(2, 5)
        for batch_shape in (torch.Size([4, 3]), torch.Size([5, 4, 3])):
            scale = torch.rand(*batch_shape) + 0.1
            op = _BroadcastInputLinearOperator(x, x, scale)
            self.assertAllClose(op.sum(0).to_dense(), op.to_dense().sum(0))

            kop = KernelLinearOperator(
                x,
                x,
                covar_func=lambda x1, x2, s: s * (x1 @ x2.mT),
                num_nonbatch_dimensions={"s": 2},
                s=scale.unsqueeze(-1).unsqueeze(-1),
            )
            self.assertAllClose(kop.sum(0).to_dense(), kop.to_dense().sum(0))

        diag = DiagLinearOperator(torch.randn(2, 3, 6))
        cat_op = torch.cat([diag, diag], dim=-2)
        self.assertEqual(cat_op.shape, torch.Size([2, 3, 12, 6]))
        self.assertAllClose(cat_op.to_dense(), torch.cat([diag.to_dense(), diag.to_dense()], dim=-2))

        empty_slice = diag[..., :0, 1:4]
        self.assertEqual(empty_slice.shape, torch.Size([2, 3, 0, 3]))
        self.assertEqual(empty_slice.to_dense().shape, torch.Size([2, 3, 0, 3]))


if __name__ == "__main__":
    unittest.main()
