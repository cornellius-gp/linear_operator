#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import DenseLinearOperator, DiagLinearOperator, MatmulLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase, RectangularLinearOperatorTestCase


class TestMatmulLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 1

    def create_linear_op(self):
        lhs = torch.randn(5, 6, requires_grad=True)
        rhs = lhs.clone().detach().mT
        covar = MatmulLinearOperator(lhs, rhs)
        return covar

    def evaluate_linear_op(self, linear_op):
        return linear_op.left_linear_op.tensor.matmul(linear_op.right_linear_op.tensor)


class TestMatmulLinearOperatorBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 3

    def create_linear_op(self):
        lhs = torch.randn(5, 5, 6, requires_grad=True)
        rhs = lhs.clone().detach().mT
        covar = MatmulLinearOperator(lhs, rhs)
        return covar

    def evaluate_linear_op(self, linear_op):
        return linear_op.left_linear_op.tensor.matmul(linear_op.right_linear_op.tensor)


class TestMatmulLinearOperatorRectangular(RectangularLinearOperatorTestCase, unittest.TestCase):
    def create_linear_op(self):
        lhs = torch.randn(5, 3, requires_grad=True)
        rhs = torch.randn(3, 6, requires_grad=True)
        covar = MatmulLinearOperator(lhs, rhs)
        return covar

    def evaluate_linear_op(self, linear_op):
        return linear_op.left_linear_op.tensor.matmul(linear_op.right_linear_op.tensor)


class TestMatmulLinearOperatorRectangularMultiBatch(RectangularLinearOperatorTestCase, unittest.TestCase):
    def create_linear_op(self):
        lhs = torch.randn(2, 3, 5, 3, requires_grad=True)
        rhs = torch.randn(2, 3, 3, 6, requires_grad=True)
        covar = MatmulLinearOperator(lhs, rhs)
        return covar

    def evaluate_linear_op(self, linear_op):
        return linear_op.left_linear_op.tensor.matmul(linear_op.right_linear_op.tensor)


class TestMatmulLinearOperatorDiagOptimization(unittest.TestCase):
    """Tests for efficient diagonal matrix multiplication in to_dense()."""

    def test_diag_left_matmul_to_dense(self):
        """Test D @ A uses element-wise multiplication."""
        diag = torch.tensor([1.0, 2.0, 3.0, 4.0])
        A = torch.randn(4, 5)

        D = DiagLinearOperator(diag)
        result = MatmulLinearOperator(D, DenseLinearOperator(A))

        expected = torch.diag(diag) @ A
        self.assertTrue(torch.allclose(result.to_dense(), expected))

    def test_diag_right_matmul_to_dense(self):
        """Test A @ D uses element-wise multiplication."""
        diag = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        A = torch.randn(4, 5)

        D = DiagLinearOperator(diag)
        result = MatmulLinearOperator(DenseLinearOperator(A), D)

        expected = A @ torch.diag(diag)
        self.assertTrue(torch.allclose(result.to_dense(), expected))

    def test_diag_sandwich_to_dense(self):
        """Test D1 @ A @ D2 uses element-wise multiplication (the main bug fix)."""
        diag1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        diag2 = torch.tensor([0.5, 1.5, 2.5, 3.5])
        A = torch.randn(4, 4)

        D1 = DiagLinearOperator(diag1)
        D2 = DiagLinearOperator(diag2)

        result = D1 @ DenseLinearOperator(A) @ D2
        expected = torch.diag(diag1) @ A @ torch.diag(diag2)
        self.assertTrue(torch.allclose(result.to_dense(), expected))

    def test_diag_sandwich_batch(self):
        """Test D1 @ A @ D2 with batch dimensions."""
        batch_size = 3
        n = 4

        diag1 = torch.randn(batch_size, n).abs()
        diag2 = torch.randn(batch_size, n).abs()
        A = torch.randn(batch_size, n, n)

        D1 = DiagLinearOperator(diag1)
        D2 = DiagLinearOperator(diag2)

        result = D1 @ DenseLinearOperator(A) @ D2
        expected = torch.diag_embed(diag1) @ A @ torch.diag_embed(diag2)
        self.assertTrue(torch.allclose(result.to_dense(), expected))


if __name__ == "__main__":
    unittest.main()
