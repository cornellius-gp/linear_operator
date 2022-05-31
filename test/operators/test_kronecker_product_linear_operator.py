#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import (
    DenseLinearOperator,
    DiagLinearOperator,
    KroneckerProductDiagLinearOperator,
    KroneckerProductLinearOperator,
)
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase, RectangularLinearOperatorTestCase

from .test_diag_linear_operator import TestDiagLinearOperator


def kron(a, b):
    res = []
    for i in range(a.size(-2)):
        row_res = []
        for j in range(a.size(-1)):
            row_res.append(b * a[..., i, j].unsqueeze(-1).unsqueeze(-2))
        res.append(torch.cat(row_res, -1))
    return torch.cat(res, -2)


def kron_diag(*lts):
    """Compute diagonal of a KroneckerProductLinearOperator from the diagonals of the constituiting tensors"""
    lead_diag = lts[0].diagonal(dim1=-1, dim2=-2)
    if len(lts) == 1:  # base case:
        return lead_diag
    trail_diag = kron_diag(*lts[1:])
    diag = lead_diag.unsqueeze(-2) * trail_diag.unsqueeze(-1)
    return diag.mT.reshape(*diag.shape[:-2], -1)


class TestKroneckerProductLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    should_call_lanczos = True
    should_call_lanczos_diagonalization = False

    def create_linear_op(self):
        a = torch.tensor([[4, 0, 2], [0, 3, -1], [2, -1, 3]], dtype=torch.float)
        b = torch.tensor([[2, 1], [1, 2]], dtype=torch.float)
        c = torch.tensor([[4, 0.5, 1, 0], [0.5, 4, -1, 0], [1, -1, 3, 0], [0, 0, 0, 4]], dtype=torch.float)
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


class TestKroneckerProductDiagLinearOperator(TestDiagLinearOperator):
    should_call_lanczos_diagonalization = False

    def create_linear_op(self):
        a = torch.tensor([4.0, 1.0, 2.0], dtype=torch.float)
        b = torch.tensor([3.0, 1.3], dtype=torch.float)
        c = torch.tensor([1.75, 1.95, 1.05, 0.25], dtype=torch.float)
        a.requires_grad_(True)
        b.requires_grad_(True)
        c.requires_grad_(True)
        kp_linear_op = KroneckerProductDiagLinearOperator(
            DiagLinearOperator(a), DiagLinearOperator(b), DiagLinearOperator(c)
        )
        return kp_linear_op

    def evaluate_linear_op(self, linear_op):
        res_diag = kron_diag(*linear_op.linear_ops)
        return torch.diag_embed(res_diag)

    def test_exp(self):
        pass

    def test_log(self):
        pass


class TestKroneckerProductLinearOperatorBatch(TestKroneckerProductLinearOperator):
    seed = 0
    should_call_lanczos = True
    should_call_lanczos_diagonalization = False

    def create_linear_op(self):
        a = torch.tensor([[4, 0, 2], [0, 3, -1], [2, -1, 3]], dtype=torch.float).repeat(3, 1, 1)
        b = torch.tensor([[2, 1], [1, 2]], dtype=torch.float).repeat(3, 1, 1)
        c = torch.tensor([[4, 0.1, 1, 0], [0.1, 4, -1, 0], [1, -1, 3, 0], [0, 0, 0, 4]], dtype=torch.float).repeat(
            3, 1, 1
        )
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


if __name__ == "__main__":
    unittest.main()
