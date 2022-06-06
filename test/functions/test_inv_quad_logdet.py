#!/usr/bin/env python3

import unittest
from unittest.mock import MagicMock, patch

import torch

import linear_operator
from linear_operator.operators import DenseLinearOperator
from linear_operator.test.base_test_case import BaseTestCase


class TestInvQuadLogDetNonBatch(BaseTestCase, unittest.TestCase):
    seed = 0
    matrix_shape = torch.Size((50, 50))

    def _test_inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, improper_logdet=False, add_diag=False):
        # Set up
        x = torch.randn(*self.__class__.matrix_shape[:-1], 3)
        ls = torch.tensor(2.0).requires_grad_(True)
        ls_clone = torch.tensor(2.0).requires_grad_(True)
        mat = (x[..., :, None, :] - x[..., None, :, :]).pow(2.0).sum(dim=-1).mul(-0.5 * ls).exp()
        mat_clone = (x[..., :, None, :] - x[..., None, :, :]).pow(2.0).sum(dim=-1).mul(-0.5 * ls_clone).exp()

        if inv_quad_rhs is not None:
            inv_quad_rhs.requires_grad_(True)
            inv_quad_rhs_clone = inv_quad_rhs.detach().clone().requires_grad_(True)

        mat_clone_with_diag = mat_clone
        if add_diag:
            mat_clone_with_diag = mat_clone_with_diag + torch.eye(mat_clone.size(-1))

        if inv_quad_rhs is not None:
            actual_inv_quad = mat_clone_with_diag.inverse().matmul(inv_quad_rhs_clone).mul(inv_quad_rhs_clone)
            actual_inv_quad = actual_inv_quad.sum([-1, -2]) if inv_quad_rhs.dim() >= 2 else actual_inv_quad.sum()
        if logdet:
            flattened_tensor = mat_clone_with_diag.view(-1, *mat_clone.shape[-2:])
            logdets = torch.cat([mat.logdet().unsqueeze(0) for mat in flattened_tensor])
            if mat_clone.dim() > 2:
                actual_logdet = logdets.view(*mat_clone.shape[:-2])
            else:
                actual_logdet = logdets.squeeze()

        # Compute values with LinearOperator
        _wrapped_cg = MagicMock(wraps=linear_operator.utils.linear_cg)
        with linear_operator.settings.num_trace_samples(2000), linear_operator.settings.max_cholesky_size(
            0
        ), linear_operator.settings.cg_tolerance(1e-5), linear_operator.settings.skip_logdet_forward(
            improper_logdet
        ), patch(
            "linear_operator.utils.linear_cg", new=_wrapped_cg
        ) as linear_cg_mock, linear_operator.settings.min_preconditioning_size(
            0
        ), linear_operator.settings.max_preconditioner_size(
            30
        ):
            linear_op = DenseLinearOperator(mat)

            if add_diag:
                linear_op = linear_op.add_jitter(1.0)

            res_inv_quad, res_logdet = linear_operator.inv_quad_logdet(
                linear_op, inv_quad_rhs=inv_quad_rhs, logdet=logdet
            )

        # Compare forward pass
        if inv_quad_rhs is not None:
            self.assertAllClose(res_inv_quad, actual_inv_quad, rtol=1e-2)
        if logdet and not improper_logdet:
            self.assertAllClose(res_logdet, actual_logdet, rtol=1e-1, atol=2e-1)

        # Backward
        if inv_quad_rhs is not None:
            actual_inv_quad.sum().backward(retain_graph=True)
            res_inv_quad.sum().backward(retain_graph=True)
        if logdet:
            actual_logdet.sum().backward()
            res_logdet.sum().backward()

        self.assertAllClose(ls.grad, ls_clone.grad, rtol=1e-2, atol=1e-2)
        if inv_quad_rhs is not None:
            self.assertAllClose(inv_quad_rhs.grad, inv_quad_rhs_clone.grad, rtol=2e-2, atol=1e-2)

        # Make sure CG was called
        self.assertTrue(linear_cg_mock.called)

    def test_inv_quad_logdet_vector(self):
        rhs = torch.randn(self.matrix_shape[-1])
        self._test_inv_quad_logdet(inv_quad_rhs=rhs, logdet=True)

    def test_precond_inv_quad_logdet_vector(self):
        rhs = torch.randn(self.matrix_shape[-1])
        self._test_inv_quad_logdet(inv_quad_rhs=rhs, logdet=True, add_diag=True)

    def test_inv_quad_only_vector(self):
        rhs = torch.randn(self.matrix_shape[-1])
        self._test_inv_quad_logdet(inv_quad_rhs=rhs, logdet=False)

    def test_precond_inv_quad_only_vector(self):
        rhs = torch.randn(self.matrix_shape[-1])
        self._test_inv_quad_logdet(inv_quad_rhs=rhs, logdet=False, add_diag=True)

    def test_inv_quad_logdet_many_vectors(self):
        rhs = torch.randn(*self.matrix_shape[:-1], 5)
        self._test_inv_quad_logdet(inv_quad_rhs=rhs, logdet=True)

    def test_precond_inv_quad_logdet_many_vectors(self):
        rhs = torch.randn(*self.matrix_shape[:-1], 5)
        self._test_inv_quad_logdet(inv_quad_rhs=rhs, logdet=True, add_diag=True)

    def test_inv_quad_logdet_many_vectors_improper(self):
        rhs = torch.randn(*self.matrix_shape[:-1], 5)
        self._test_inv_quad_logdet(inv_quad_rhs=rhs, logdet=True, improper_logdet=True)

    def test_precond_inv_quad_logdet_many_vectors_improper(self):
        rhs = torch.randn(*self.matrix_shape[:-1], 5)
        self._test_inv_quad_logdet(inv_quad_rhs=rhs, logdet=True, improper_logdet=True, add_diag=True)

    def test_inv_quad_only_many_vectors(self):
        rhs = torch.randn(*self.matrix_shape[:-1], 5)
        self._test_inv_quad_logdet(inv_quad_rhs=rhs, logdet=False)

    def test_precond_inv_quad_only_many_vectors(self):
        rhs = torch.randn(*self.matrix_shape[:-1], 5)
        self._test_inv_quad_logdet(inv_quad_rhs=rhs, logdet=False, add_diag=True)


class TestInvQuadLogDetBatch(TestInvQuadLogDetNonBatch):
    seed = 0
    matrix_shape = torch.Size((3, 50, 50))

    def test_inv_quad_logdet_vector(self):
        pass

    def test_precond_inv_quad_logdet_vector(self):
        pass

    def test_inv_quad_only_vector(self):
        pass

    def test_precond_inv_quad_only_vector(self):
        pass


class TestInvQuadLogDetMultiBatch(TestInvQuadLogDetBatch):
    seed = 0
    matrix_shape = torch.Size((2, 3, 50, 50))


if __name__ == "__main__":
    unittest.main()
