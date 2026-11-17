#!/usr/bin/env python3

import unittest

import torch

import linear_operator.utils.toeplitz as toeplitz
from linear_operator.operators import ToeplitzLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class TestToeplitzLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 1

    def test_hadamard_product(self):
        a = torch.randn(2, 1, 5, dtype=torch.double, requires_grad=True)
        b = torch.randn(3, 5, dtype=torch.double, requires_grad=True)
        rhs = torch.randn(5, 2, dtype=torch.double)
        distance = (torch.arange(5)[:, None] - torch.arange(5)).abs()
        expected = (a[..., distance] * b[..., distance]) @ rhs
        product = ToeplitzLinearOperator(a) * ToeplitzLinearOperator(b)
        self.assertIsInstance(product, ToeplitzLinearOperator)
        actual = product @ rhs
        torch.testing.assert_close(actual, expected)
        grad = torch.randn_like(actual)
        for observed, reference in zip(
            torch.autograd.grad(actual, (a, b), grad), torch.autograd.grad(expected, (a, b), grad)
        ):
            torch.testing.assert_close(observed, reference)

    def create_linear_op(self):
        toeplitz_column = torch.tensor([4, 0.5, 0, 1], dtype=torch.float, requires_grad=True)
        return ToeplitzLinearOperator(toeplitz_column)

    def evaluate_linear_op(self, linear_op):
        return toeplitz.sym_toeplitz(linear_op.column)


class TestToeplitzLinearOperatorBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_op(self):
        toeplitz_column = torch.tensor([[2, -1, 0.5, 0.25], [4, 0.5, 0, 1]], dtype=torch.float, requires_grad=True)
        return ToeplitzLinearOperator(toeplitz_column)

    def evaluate_linear_op(self, linear_op):
        return torch.cat(
            [
                toeplitz.sym_toeplitz(linear_op.column[0]).unsqueeze(0),
                toeplitz.sym_toeplitz(linear_op.column[1]).unsqueeze(0),
            ]
        )


class TestToeplitzLinearOperatorMultiBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_op(self):
        toeplitz_column = torch.tensor([[2, -1, 0.5, 0.25], [4, 0.5, 0, 1]], dtype=torch.float)
        toeplitz_column = toeplitz_column.repeat(3, 1, 1)
        toeplitz_column.requires_grad_(True)
        return ToeplitzLinearOperator(toeplitz_column)

    def evaluate_linear_op(self, linear_op):
        return torch.cat(
            [
                toeplitz.sym_toeplitz(linear_op.column[0, 0]).unsqueeze(0),
                toeplitz.sym_toeplitz(linear_op.column[0, 1]).unsqueeze(0),
                toeplitz.sym_toeplitz(linear_op.column[1, 0]).unsqueeze(0),
                toeplitz.sym_toeplitz(linear_op.column[1, 1]).unsqueeze(0),
                toeplitz.sym_toeplitz(linear_op.column[2, 0]).unsqueeze(0),
                toeplitz.sym_toeplitz(linear_op.column[2, 1]).unsqueeze(0),
            ]
        ).view(3, 2, 4, 4)


if __name__ == "__main__":
    unittest.main()
