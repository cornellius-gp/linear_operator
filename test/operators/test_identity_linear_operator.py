#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import IdentityLinearOperator, to_dense
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class TestIdentityLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    def _test_matmul(self, rhs):
        linear_op = self.create_linear_op().detach().requires_grad_(True)
        linear_op_copy = linear_op.clone().detach().requires_grad_(True)
        evaluated = self.evaluate_linear_op(linear_op_copy)
        rhs_evaluated = to_dense(rhs)

        res = linear_op.matmul(rhs)
        actual = evaluated.matmul(rhs_evaluated)
        res_evaluated = to_dense(res)
        self.assertAllClose(res_evaluated, actual)

    def _test_rmatmul(self, lhs):
        linear_op = self.create_linear_op().detach().requires_grad_(True)
        linear_op_copy = linear_op.clone().detach().requires_grad_(True)
        evaluated = self.evaluate_linear_op(linear_op_copy)

        res = lhs @ linear_op
        actual = lhs @ evaluated
        self.assertAllClose(res, actual)

    def _test_solve(self, rhs, lhs=None, cholesky=False):
        linear_op = self.create_linear_op().detach().requires_grad_(True)
        linear_op_copy = linear_op.clone().detach().requires_grad_(True)
        evaluated = self.evaluate_linear_op(linear_op_copy)

        # Create a test right hand side and left hand side
        rhs.requires_grad_(True)
        rhs_copy = rhs.clone().detach().requires_grad_(True)
        if lhs is not None:
            lhs.requires_grad_(True)
            lhs_copy = lhs.clone().detach().requires_grad_(True)
        if lhs is not None:
            res = linear_op.solve(rhs, lhs)
            actual = lhs_copy @ evaluated.inverse() @ rhs_copy
        else:
            res = linear_op.solve(rhs)
            actual = evaluated.inverse().matmul(rhs_copy)
        self.assertAllClose(res, actual, **self.tolerances["solve"])

    def _test_inv_quad_logdet(self, reduce_inv_quad=True, cholesky=False, linear_op=None):
        if linear_op is None:
            linear_op = self.create_linear_op()
        evaluated = self.evaluate_linear_op(linear_op)
        flattened_evaluated = evaluated.view(-1, *linear_op.matrix_shape)

        vecs = torch.randn(*linear_op.batch_shape, linear_op.size(-1), 3, requires_grad=True)
        vecs_copy = vecs.clone().detach().requires_grad_(True)
        res_inv_quad, res_logdet = linear_op.inv_quad_logdet(
            inv_quad_rhs=vecs, logdet=True, reduce_inv_quad=reduce_inv_quad
        )

        actual_inv_quad = evaluated.inverse().matmul(vecs_copy).mul(vecs_copy).sum(-2)
        if reduce_inv_quad:
            actual_inv_quad = actual_inv_quad.sum(-1)
        actual_logdet = torch.cat(
            [torch.logdet(flattened_evaluated[i]).unsqueeze(0) for i in range(linear_op.batch_shape.numel())]
        ).view(linear_op.batch_shape)

        self.assertAllClose(res_inv_quad, actual_inv_quad, **self.tolerances["inv_quad"])
        self.assertAllClose(res_logdet, actual_logdet, **self.tolerances["logdet"])

    def create_linear_op(self):
        return IdentityLinearOperator(5)

    def evaluate_linear_op(self, linear_op):
        return torch.eye(5)

    def test_diagonalization(self, symeig=False):
        linear_op = self.create_linear_op()
        evals, evecs = linear_op.diagonalization()
        self.assertAllClose(evals, torch.ones(linear_op.shape[:-1]))
        self.assertAllClose(evecs.to_dense(), torch.eye(linear_op.size(-1)).expand(linear_op.shape))

    def test_eigh(self):
        linear_op = self.create_linear_op()
        evals, evecs = linear_op.eigh()
        self.assertAllClose(evals, torch.ones(linear_op.shape[:-1]))
        self.assertAllClose(evecs.to_dense(), torch.eye(linear_op.size(-1)).expand(linear_op.shape))

    def test_eigvalsh(self):
        linear_op = self.create_linear_op()
        evals = linear_op.eigvalsh()
        self.assertAllClose(evals, torch.ones(linear_op.shape[:-1]))

    def test_exp(self):
        linear_op = self.create_linear_op()
        exp = linear_op.exp().to_dense()
        self.assertAllClose(exp, torch.eye(linear_op.size(-1)).expand(*linear_op.shape))

    def test_log(self):
        linear_op = self.create_linear_op()
        log = linear_op.log().to_dense()
        self.assertAllClose(log, torch.zeros(*linear_op.shape))

    def test_logdet(self):
        linear_op = self.create_linear_op()
        self.assertAllClose(torch.logdet(linear_op), torch.zeros(linear_op.batch_shape))

    def test_sqrt_inv_matmul(self):
        linear_op = self.create_linear_op().detach().requires_grad_(True)
        if len(linear_op.batch_shape):
            return

        linear_op_copy = linear_op.clone().detach().requires_grad_(True)
        evaluated = self.evaluate_linear_op(linear_op_copy)

        # Create a test right hand side and left hand side
        rhs = torch.randn(*linear_op.shape[:-1], 3).requires_grad_(True)
        lhs = torch.randn(*linear_op.shape[:-2], 2, linear_op.size(-1)).requires_grad_(True)
        rhs_copy = rhs.clone().detach().requires_grad_(True)
        lhs_copy = lhs.clone().detach().requires_grad_(True)

        # Perform forward pass
        sqrt_inv_matmul_res, inv_quad_res = linear_op.sqrt_inv_matmul(rhs, lhs)
        evals, evecs = torch.linalg.eigh(evaluated)
        matrix_inv_root = evecs @ (evals.sqrt().reciprocal().unsqueeze(-1) * evecs.mT)
        sqrt_inv_matmul_actual = lhs_copy @ matrix_inv_root @ rhs_copy
        inv_quad_actual = (lhs_copy @ matrix_inv_root).pow(2).sum(dim=-1)

        # Check forward pass
        self.assertAllClose(sqrt_inv_matmul_res, sqrt_inv_matmul_actual, **self.tolerances["sqrt_inv_matmul"])
        self.assertAllClose(inv_quad_res, inv_quad_actual, **self.tolerances["sqrt_inv_matmul"])

    def test_sqrt_inv_matmul_no_lhs(self):
        linear_op = self.create_linear_op().detach().requires_grad_(True)
        if len(linear_op.batch_shape):
            return

        linear_op_copy = linear_op.clone().detach().requires_grad_(True)
        evaluated = self.evaluate_linear_op(linear_op_copy)

        # Create a test right hand side and left hand side
        rhs = torch.randn(*linear_op.shape[:-1], 3).requires_grad_(True)
        rhs_copy = rhs.clone().detach().requires_grad_(True)

        # Perform forward pass
        sqrt_inv_matmul_res = linear_op.sqrt_inv_matmul(rhs)
        evals, evecs = torch.linalg.eigh(evaluated)
        matrix_inv_root = evecs @ (evals.sqrt().reciprocal().unsqueeze(-1) * evecs.mT)
        sqrt_inv_matmul_actual = matrix_inv_root @ rhs_copy

        # Check forward pass
        self.assertAllClose(sqrt_inv_matmul_res, sqrt_inv_matmul_actual, **self.tolerances["sqrt_inv_matmul"])

    def test_root_decomposition(self, cholesky=False):
        linear_op = self.create_linear_op()
        root_decomp = linear_op.root_decomposition().root
        self.assertAllClose(root_decomp.to_dense(), torch.eye(linear_op.size(-1)).expand(linear_op.shape))

    def test_svd(self):
        linear_op = self.create_linear_op()
        U, S, Vt = linear_op.svd()
        self.assertAllClose(S, torch.ones(linear_op.shape[:-1]))
        self.assertAllClose(U.to_dense(), torch.eye(linear_op.size(-1)).expand(linear_op.shape))
        self.assertAllClose(Vt.to_dense(), torch.eye(linear_op.size(-1)).expand(linear_op.shape))

    def test_solve_triangular(self):
        linear_op = self.create_linear_op()
        rhs = torch.randn(linear_op.size(-1))
        res = torch.linalg.solve_triangular(linear_op, rhs, upper=False)
        res_actual = rhs / linear_op.diag_values
        self.assertAllClose(res, res_actual)


class TestIdentityLinearOperatorBatch(TestIdentityLinearOperator):
    def create_linear_op(self):
        return IdentityLinearOperator(5, batch_shape=torch.Size([3, 6]))

    def evaluate_linear_op(self, linear_op):
        return torch.eye(5).expand(3, 6, 5, 5)


if __name__ == "__main__":
    unittest.main()
