#!/usr/bin/env python3

import unittest
from unittest.mock import patch

import torch

from linear_operator.operators import DenseLinearOperator, KroneckerProductLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase, RectangularLinearOperatorTestCase


def kron(a, b):
    res = []
    for i in range(a.size(-2)):
        row_res = []
        for j in range(a.size(-1)):
            row_res.append(b * a[..., i, j].unsqueeze(-1).unsqueeze(-2))
        res.append(torch.cat(row_res, -1))
    return torch.cat(res, -2)


class TestKroneckerProductLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    should_call_lanczos = True
    should_call_lanczos_diagonalization = False

    def create_linear_op(self):
        a = torch.tensor([[4, 0, 2], [0, 3, -1], [2, -1, 3]], dtype=torch.float)
        b = torch.tensor([[2, 1], [1, 2]], dtype=torch.float)
        c = torch.tensor(
            [[4, 0.5, 1, 0], [0.5, 4, -1, 0], [1, -1, 3, 0], [0, 0, 0, 4]],
            dtype=torch.float,
        )
        a.requires_grad_(True)
        b.requires_grad_(True)
        c.requires_grad_(True)
        kp_linear_op = KroneckerProductLinearOperator(
            DenseLinearOperator(a), DenseLinearOperator(b), DenseLinearOperator(c)
        )
        return kp_linear_op

    def evaluate_linear_op(self, linear_op):
        res = kron(linear_op.linear_ops[0].tensor, linear_op.linear_ops[1].tensor)
        res = kron(res, linear_op.linear_ops[2].tensor)
        return res


class TestKroneckerProductLinearOperatorBatch(TestKroneckerProductLinearOperator):
    seed = 0
    should_call_lanczos = True
    should_call_lanczos_diagonalization = False

    def create_linear_op(self):
        a = torch.tensor([[4, 0, 2], [0, 3, -1], [2, -1, 3]], dtype=torch.float).repeat(3, 1, 1)
        b = torch.tensor([[2, 1], [1, 2]], dtype=torch.float).repeat(3, 1, 1)
        c = torch.tensor(
            [[4, 0.1, 1, 0], [0.1, 4, -1, 0], [1, -1, 3, 0], [0, 0, 0, 4]],
            dtype=torch.float,
        ).repeat(3, 1, 1)
        a.requires_grad_(True)
        b.requires_grad_(True)
        c.requires_grad_(True)
        kp_linear_op = KroneckerProductLinearOperator(
            DenseLinearOperator(a), DenseLinearOperator(b), DenseLinearOperator(c)
        )
        return kp_linear_op


class TestKroneckerProductLinearOperatorRectangular(RectangularLinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_op(self):
        a = torch.randn(2, 3, requires_grad=True)
        b = torch.randn(5, 2, requires_grad=True)
        c = torch.randn(6, 4, requires_grad=True)
        kp_linear_op = KroneckerProductLinearOperator(
            DenseLinearOperator(a), DenseLinearOperator(b), DenseLinearOperator(c)
        )
        return kp_linear_op

    def evaluate_linear_op(self, linear_op):
        res = kron(linear_op.linear_ops[0].tensor, linear_op.linear_ops[1].tensor)
        res = kron(res, linear_op.linear_ops[2].tensor)
        return res


class TestKroneckerProductLinearOperatorRectangularMultiBatch(TestKroneckerProductLinearOperatorRectangular):
    seed = 0

    def create_linear_op(self):
        a = torch.randn(3, 4, 2, 3, requires_grad=True)
        b = torch.randn(3, 4, 5, 2, requires_grad=True)
        c = torch.randn(3, 4, 6, 4, requires_grad=True)
        kp_linear_op = KroneckerProductLinearOperator(
            DenseLinearOperator(a), DenseLinearOperator(b), DenseLinearOperator(c)
        )
        return kp_linear_op


class TestKroneckerProductRectangularFactors(unittest.TestCase):
    def test_diagonal_empty(self):
        for dtype in (torch.float32, torch.float64):
            for shapes, expected_shape in (
                (((0, 1), (1, 0)), (0,)),
                (((1, 0), (0, 1)), (0,)),
                (((2, 1, 0, 1), (1, 3, 1, 0)), (2, 3, 0)),
            ):
                with self.subTest(dtype=dtype, shapes=shapes):
                    linear_op = KroneckerProductLinearOperator(*(torch.empty(shape, dtype=dtype) for shape in shapes))
                    torch.testing.assert_close(linear_op.diagonal(), torch.empty(expected_shape, dtype=dtype))

    def test_diagonal(self):
        for dtype in (torch.float32, torch.float64):
            for shapes in (((2, 3), (3, 2)), ((3, 2), (2, 3)), ((2, 3), (2, 2), (3, 2))):
                with self.subTest(dtype=dtype, shapes=shapes):
                    factors = [
                        torch.arange(1, rows * cols + 1, dtype=dtype).reshape(rows, cols).requires_grad_()
                        for rows, cols in shapes
                    ]
                    linear_op = KroneckerProductLinearOperator(*factors)
                    expected = factors[0]
                    for factor in factors[1:]:
                        expected = torch.kron(expected, factor)
                    expected = expected.diagonal()

                    with patch.object(KroneckerProductLinearOperator, "to_dense", side_effect=AssertionError):
                        actual = linear_op.diagonal()
                    torch.testing.assert_close(actual, expected)
                    actual_grads = torch.autograd.grad(actual.sum(), factors)
                    expected_grads = torch.autograd.grad(expected.sum(), factors)
                    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                        torch.testing.assert_close(actual_grad, expected_grad)

    def test_diagonal_broadcast_batches(self):
        a = torch.arange(1, 13, dtype=torch.float64).reshape(2, 1, 2, 3).requires_grad_()
        b = torch.arange(1, 19, dtype=torch.float64).reshape(1, 3, 3, 2).requires_grad_()
        linear_op = KroneckerProductLinearOperator(a, b)
        expected = torch.stack([torch.stack([torch.kron(ai[0], bj) for bj in b[0]]) for ai in a])
        expected = expected.diagonal(dim1=-2, dim2=-1)

        with patch.object(KroneckerProductLinearOperator, "to_dense", side_effect=AssertionError):
            actual = linear_op.diagonal()
        torch.testing.assert_close(actual, expected)
        actual_grads = torch.autograd.grad(actual.sum(), (a, b))
        expected_grads = torch.autograd.grad(expected.sum(), (a, b))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)


if __name__ == "__main__":
    unittest.main()
