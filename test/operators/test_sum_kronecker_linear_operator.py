#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import DenseLinearOperator, KroneckerProductLinearOperator, SumKroneckerLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


def kron(a, b):
    res = []
    for i in range(a.size(-2)):
        row_res = []
        for j in range(a.size(-1)):
            row_res.append(b * a[..., i, j].unsqueeze(-1).unsqueeze(-2))
        res.append(torch.cat(row_res, -1))
    return torch.cat(res, -2)


class TestSumKroneckerLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    should_call_lanczos = True
    should_call_cg = False
    skip_slq_tests = False

    def create_linear_op(self):
        a = torch.tensor([[4, 0, 2], [0, 3, -1], [2, -1, 3]], dtype=torch.float)
        b = torch.tensor([[2, 1], [1, 2]], dtype=torch.float)
        c = torch.tensor([[4, 0.5, 1], [0.5, 4, -1], [1, -1, 3]], dtype=torch.float)
        d = torch.tensor([[1.2, 0.75], [0.75, 1.2]], dtype=torch.float)

        a.requires_grad_(True)
        b.requires_grad_(True)
        c.requires_grad_(True)
        d.requires_grad_(True)
        kp_lt_1 = KroneckerProductLinearOperator(DenseLinearOperator(a), DenseLinearOperator(b))
        kp_lt_2 = KroneckerProductLinearOperator(DenseLinearOperator(c), DenseLinearOperator(d))

        return SumKroneckerLinearOperator(kp_lt_1, kp_lt_2)

    def evaluate_linear_op(self, linear_op):
        res1 = kron(linear_op.linear_ops[0].linear_ops[0].tensor, linear_op.linear_ops[0].linear_ops[1].tensor)
        res2 = kron(linear_op.linear_ops[1].linear_ops[0].tensor, linear_op.linear_ops[1].linear_ops[1].tensor)
        return res1 + res2
