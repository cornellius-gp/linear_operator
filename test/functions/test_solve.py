#!/usr/bin/env python3

import unittest

import torch

from linear_operator import settings
from linear_operator.operators import DenseLinearOperator
from linear_operator.test.base_test_case import BaseTestCase


def _ensure_symmetric_grad(grad):
    """
    A gradient-hook hack to ensure that symmetric matrix gradients are symmetric
    """
    res = torch.add(grad, grad.mT).mul(0.5)
    return res


class TestSolveNonBatch(BaseTestCase, unittest.TestCase):
    seed = 0

    def _create_mat(self):
        mat = torch.randn(8, 8)
        mat = mat @ mat.mT
        return mat

    def test_solve_vec(self):
        mat = self._create_mat().detach().requires_grad_(True)
        if mat.dim() > 2:  # This isn't a feature for batch mode
            return
        mat_copy = mat.detach().clone().requires_grad_(True)
        mat_copy.register_hook(_ensure_symmetric_grad)
        vec = torch.randn(mat.size(-1)).detach().requires_grad_(True)
        vec_copy = vec.detach().clone().requires_grad_(True)

        # Forward
        with settings.terminate_cg_by_size(False):
            res = DenseLinearOperator(mat).solve(vec)
            actual = mat_copy.inverse().matmul(vec_copy)
            self.assertAllClose(res, actual)

            # Backward
            grad_output = torch.randn_like(vec)
            res.backward(gradient=grad_output)
            actual.backward(gradient=grad_output)
            self.assertAllClose(mat.grad, mat_copy.grad)
            self.assertAllClose(vec.grad, vec_copy.grad)

    def test_solve_multiple_vecs(self):
        mat = self._create_mat().detach().requires_grad_(True)
        mat_copy = mat.detach().clone().requires_grad_(True)
        mat_copy.register_hook(_ensure_symmetric_grad)
        vecs = torch.randn(*mat.shape[:-2], mat.size(-1), 4).detach().requires_grad_(True)
        vecs_copy = vecs.detach().clone().requires_grad_(True)

        # Forward
        with settings.terminate_cg_by_size(False):
            res = DenseLinearOperator(mat).solve(vecs)
            actual = mat_copy.inverse().matmul(vecs_copy)
            self.assertAllClose(res, actual)

            # Backward
            grad_output = torch.randn_like(vecs)
            res.backward(gradient=grad_output)
            actual.backward(gradient=grad_output)
            self.assertAllClose(mat.grad, mat_copy.grad)
            self.assertAllClose(vecs.grad, vecs_copy.grad)


class TestSolveBatch(TestSolveNonBatch):
    seed = 0

    def _create_mat(self):
        mats = torch.randn(2, 8, 8)
        mats = mats @ mats.mT
        return mats


if __name__ == "__main__":
    unittest.main()
