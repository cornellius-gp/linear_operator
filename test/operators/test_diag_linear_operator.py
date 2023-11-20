#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import DiagLinearOperator, KroneckerProductDiagLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


def kron_diag(*lts):
    """Compute diagonal of a KroneckerProductLinearOperator from the diagonals of the constituiting tensors"""
    lead_diag = lts[0].diagonal(dim1=-1, dim2=-2)
    if len(lts) == 1:  # base case:
        return lead_diag
    trail_diag = kron_diag(*lts[1:])
    diag = lead_diag.unsqueeze(-2) * trail_diag.unsqueeze(-1)
    return diag.mT.reshape(*diag.shape[:-2], -1)


class TestDiagLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True
    should_call_cg = False
    should_call_lanczos = False

    def create_linear_op(self):
        diag = torch.tensor([1.0, 2.0, 4.0, 5.0, 3.0], requires_grad=True)
        return DiagLinearOperator(diag)

    def evaluate_linear_op(self, linear_op):
        diag = linear_op._diag
        return torch.diag_embed(diag)

    def test_abs(self):
        linear_op = self.create_linear_op()
        linear_op_copy = linear_op.detach().clone()
        evaluated = self.evaluate_linear_op(linear_op_copy)
        self.assertAllClose(torch.abs(linear_op).to_dense(), torch.abs(evaluated))

    def test_exp(self):
        linear_op = self.create_linear_op()
        linear_op_copy = linear_op.detach().clone()
        evaluated = self.evaluate_linear_op(linear_op_copy)
        self.assertAllClose(
            torch.exp(linear_op).diagonal(dim1=-1, dim2=-2),
            torch.exp(evaluated.diagonal(dim1=-1, dim2=-2)),
        )

    def test_inverse(self):
        linear_op = self.create_linear_op()
        linear_op_copy = linear_op.detach().clone()
        linear_op.requires_grad_(True)
        linear_op_copy.requires_grad_(True)
        evaluated = self.evaluate_linear_op(linear_op_copy)

        inverse = torch.inverse(linear_op).to_dense()
        inverse_actual = evaluated.inverse()
        self.assertAllClose(inverse, inverse_actual)

        # Backwards
        inverse.sum().backward()
        inverse_actual.sum().backward()

        # Check grads
        for arg, arg_copy in zip(linear_op.representation(), linear_op_copy.representation()):
            if arg_copy.requires_grad and arg_copy.is_leaf and arg_copy.grad is not None:
                self.assertAllClose(arg.grad, arg_copy.grad)

    def test_log(self):
        linear_op = self.create_linear_op()
        linear_op_copy = linear_op.detach().clone()
        evaluated = self.evaluate_linear_op(linear_op_copy)
        self.assertAllClose(
            torch.log(linear_op).diagonal(dim1=-1, dim2=-2),
            torch.log(evaluated.diagonal(dim1=-1, dim2=-2)),
        )

    def test_solve_triangular(self):
        linear_op = self.create_linear_op()
        rhs = torch.randn(linear_op.size(-1))
        res = torch.linalg.solve_triangular(linear_op, rhs, upper=False)
        res_actual = rhs / linear_op.diagonal()
        self.assertAllClose(res, res_actual)
        res = torch.linalg.solve_triangular(linear_op, rhs, upper=True)
        res_actual = rhs / linear_op.diagonal()
        self.assertAllClose(res, res_actual)
        # unittriangular case
        with self.assertRaisesRegex(RuntimeError, "Received `unitriangular=True`"):
            torch.linalg.solve_triangular(linear_op, rhs, upper=False, unitriangular=True)
        linear_op = DiagLinearOperator(torch.ones(4))  # TODO: Test gradients
        res = torch.linalg.solve_triangular(linear_op, rhs, upper=False, unitriangular=True)
        self.assertAllClose(res, rhs)

    def test_sqrt(self):
        linear_op = self.create_linear_op()
        linear_op_copy = linear_op.detach().clone()
        evaluated = self.evaluate_linear_op(linear_op_copy)
        self.assertAllClose(torch.sqrt(linear_op).to_dense(), torch.sqrt(evaluated))


class TestDiagLinearOperatorBatch(TestDiagLinearOperator):
    seed = 0

    def create_linear_op(self):
        diag = torch.tensor(
            [
                [1.0, 2.0, 4.0, 2.0, 3.0],
                [2.0, 1.0, 2.0, 1.0, 4.0],
                [1.0, 2.0, 2.0, 3.0, 4.0],
            ],
            requires_grad=True,
        )
        return DiagLinearOperator(diag)

    def evaluate_linear_op(self, linear_op):
        diag = linear_op._diag
        return torch.diag_embed(diag)


class TestDiagLinearOperatorMultiBatch(TestDiagLinearOperator):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = True
    skip_slq_tests = True

    def create_linear_op(self):
        diag = torch.randn(6, 3, 5).pow_(2)
        diag.requires_grad_(True)
        return DiagLinearOperator(diag)

    def evaluate_linear_op(self, linear_op):
        diag = linear_op._diag
        return torch.diag_embed(diag)


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


if __name__ == "__main__":
    unittest.main()
