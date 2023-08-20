#!/usr/bin/env python3

import unittest

import torch

from linear_operator import to_linear_operator
from linear_operator.operators.masked_linear_operator import MaskedLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase, RectangularLinearOperatorTestCase


class TestMaskedLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 2023

    def create_linear_op(self):
        base = torch.randn(5, 5)
        base = base.mT @ base
        base.requires_grad_(True)
        base = to_linear_operator(base)
        mask = torch.tensor([1, 1, 0, 1, 0], dtype=torch.bool)
        covar = MaskedLinearOperator(base, mask, mask)
        return covar

    def evaluate_linear_op(self, linear_op):
        base = linear_op.base.to_dense()
        return base[..., linear_op.row_mask, :][..., linear_op.col_mask]

    def test_to_double(self):
        # overwrite the test_to_double under LinearOperatorTestCase.
        # specifically check if the mask matrices are still boolean after conversion.
        linear_op = self.create_linear_op()
        try:
            linear_op = linear_op.to(torch.float64)
            linear_op.numpy()
        except RuntimeError:
            raise RuntimeError(f"Could not convert {type(linear_op)} to double.")
        self.assertEqual(linear_op.dtype, torch.float64)
        self.assertEqual(linear_op.col_mask.dtype, torch.bool)
        self.assertEqual(linear_op.row_mask.dtype, torch.bool)


class TestMaskedLinearOperatorBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 2023

    def create_linear_op(self):
        base = torch.randn(2, 5, 5)
        base = base.mT @ base
        base.requires_grad_(True)
        base = to_linear_operator(base)
        mask = torch.tensor([1, 1, 0, 1, 0], dtype=torch.bool)
        covar = MaskedLinearOperator(base, mask, mask)
        return covar

    def evaluate_linear_op(self, linear_op):
        base = linear_op.base.to_dense()
        return base[..., linear_op.row_mask, :][..., linear_op.col_mask]


class TestMaskedLinearOperatorRectangular(RectangularLinearOperatorTestCase, unittest.TestCase):
    seed = 2023

    def create_linear_op(self):
        base = to_linear_operator(torch.randn(5, 6, requires_grad=True))
        row_mask = torch.tensor([1, 1, 0, 1, 0], dtype=torch.bool)
        col_mask = torch.tensor([1, 0, 1, 0, 1, 1], dtype=torch.bool)
        covar = MaskedLinearOperator(base, row_mask, col_mask)
        return covar

    def evaluate_linear_op(self, linear_op):
        base = linear_op.base.to_dense()
        return base[..., linear_op.row_mask, :][..., linear_op.col_mask]


class TestMaskedLinearOperatorRectangularMultiBatch(RectangularLinearOperatorTestCase, unittest.TestCase):
    seed = 2023

    def create_linear_op(self):
        base = to_linear_operator(torch.randn(2, 3, 5, 6, requires_grad=True))
        row_mask = torch.tensor([1, 1, 0, 1, 0], dtype=torch.bool)
        col_mask = torch.tensor([1, 0, 1, 0, 1, 1], dtype=torch.bool)
        covar = MaskedLinearOperator(base, row_mask, col_mask)
        return covar

    def evaluate_linear_op(self, linear_op):
        base = linear_op.base.to_dense()
        return base[..., linear_op.row_mask, :][..., linear_op.col_mask]


if __name__ == "__main__":
    unittest.main()
