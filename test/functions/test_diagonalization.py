#!/usr/bin/env python3

import unittest

import torch

import linear_operator
from linear_operator.test.base_test_case import BaseTestCase


class TestDiagonalization(BaseTestCase, unittest.TestCase):
    seed = 0

    def _create_mat(self):
        mat = torch.randn(4, 4)
        mat = mat @ mat.mT
        mat.div_(5).add_(torch.eye(4))
        return mat

    def test_diagonalization(self):
        mat = self._create_mat().detach().requires_grad_(True)
        mat_clone = mat.detach().clone().requires_grad_(True)

        for method in ["symeig", "lanczos"]:
            # Forward
            evals, evecs = linear_operator.diagonalization(mat, method=method)
            evecs = evecs.to_dense()
            res = evecs.matmul(torch.diag_embed(evals)).matmul(evecs.mT)
            self.assertAllClose(res, mat)

            # Backward
            sum([mat.trace() for mat in res.view(-1, mat.size(-2), mat.size(-1))]).backward()
            sum([mat.trace() for mat in mat_clone.view(-1, mat.size(-2), mat.size(-1))]).backward()
            self.assertAllClose(mat.grad, mat_clone.grad)


class TestDiagonalizationBatch(TestDiagonalization):
    seed = 0

    def _create_mat(self):
        mat = torch.randn(3, 4, 4)
        mat = mat @ mat.mT
        mat.div_(5).add_(torch.eye(4).unsqueeze_(0))
        return mat


class TestDiagonalizationMultiBatch(TestDiagonalization):
    seed = 0

    def _create_mat(self):
        mat = torch.randn(2, 3, 4, 4)
        mat = mat @ mat.mT
        mat.div_(5).add_(torch.eye(4).unsqueeze_(0))
        return mat


if __name__ == "__main__":
    unittest.main()
