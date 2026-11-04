#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import BlockDiagLinearOperator, DenseLinearOperator, DiagLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class TestBlockDiagLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    # Whether to initialize `BlockDiagLinearOperator` from a tensor or a linear operator.
    _initialize_from_tensor = False

    def create_linear_op(self):
        blocks = torch.randn(8, 4, 4)
        blocks = blocks.matmul(blocks.mT)
        blocks.add_(torch.eye(4, 4).unsqueeze_(0))

        return (
            BlockDiagLinearOperator(blocks)
            if self._initialize_from_tensor
            else BlockDiagLinearOperator(DenseLinearOperator(blocks))
        )

    def evaluate_linear_op(self, linear_op):
        blocks = linear_op.base_linear_op.tensor
        actual = torch.zeros(32, 32)
        for i in range(8):
            actual[i * 4 : (i + 1) * 4, i * 4 : (i + 1) * 4] = blocks[i]
        return actual


class TestBlockDiagLinearOperatorFromTensor(TestBlockDiagLinearOperator):
    _initialize_from_tensor = True


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
        for base_op, test_name in zip(base_operators, subtest_names):
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


class TestBlockDiagDiagonalMatmulBroadcast(unittest.TestCase):
    def test_diagonal_matmul_broadcast(self) -> None:
        batches = (((), (2,)), ((2,), ()), ((2, 1), (1, 2)), ((1, 2), (2, 1)), ((2,), (2,)))
        for block_batch, diag_batch in batches:
            for dtype in (torch.float32, torch.float64):
                shape = (*block_batch, 3, 2, 2)
                root = torch.arange(torch.Size(shape).numel(), dtype=dtype).reshape(shape) / 100
                blocks = (root @ root.mT + torch.eye(2, dtype=dtype)).requires_grad_()
                diagonal = torch.linspace(0.7, 1.6, torch.Size((*diag_batch, 6)).numel(), dtype=dtype)
                diagonal = diagonal.reshape(*diag_batch, 6).requires_grad_()
                dense = torch.zeros(*block_batch, 6, 6, dtype=dtype)
                for i in range(3):
                    dense[..., 2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = blocks[..., i, :, :]
                for diagonal_left in (False, True):
                    with self.subTest(batch=(block_batch, diag_batch), dtype=dtype, diagonal_left=diagonal_left):
                        block_op = BlockDiagLinearOperator(DenseLinearOperator(blocks))
                        diag_op = DiagLinearOperator(diagonal)
                        actual = diag_op @ block_op if diagonal_left else block_op @ diag_op
                        expected = diagonal.unsqueeze(-1) * dense if diagonal_left else dense * diagonal.unsqueeze(-2)
                        self.assertEqual(actual.shape, expected.shape)
                        torch.testing.assert_close(actual.to_dense(), expected)
                        torch.testing.assert_close(actual.diagonal(), expected.diagonal(dim1=-2, dim2=-1))
                        weight = torch.linspace(-1, 1, expected.numel(), dtype=dtype).reshape(expected.shape)
                        actual_grads = torch.autograd.grad(
                            (actual.to_dense() * weight).sum(), (blocks, diagonal), retain_graph=True
                        )
                        expected_grads = torch.autograd.grad(
                            (expected * weight).sum(), (blocks, diagonal), retain_graph=True
                        )
                        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                            torch.testing.assert_close(actual_grad, expected_grad)


if __name__ == "__main__":
    unittest.main()
