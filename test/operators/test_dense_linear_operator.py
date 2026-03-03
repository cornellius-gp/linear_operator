#!/usr/bin/env python3

import unittest
from unittest.mock import patch

import torch

import linear_operator
from linear_operator.operators import DenseLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class TestDenseLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_op(self):
        mat = torch.randn(5, 6)
        mat = mat.matmul(mat.mT)
        mat.requires_grad_(True)
        return DenseLinearOperator(mat)

    def evaluate_linear_op(self, linear_op):
        return linear_op.tensor

    def test_root_decomposition_exact(self):
        linear_op = self.create_linear_op()
        test_mat = torch.randn(*linear_op.batch_shape, linear_op.size(-1), 5)
        with linear_operator.settings.fast_computations(covar_root_decomposition=False):
            root_approx = linear_op.root_decomposition()
            res = root_approx.matmul(test_mat)
            actual = linear_op.matmul(test_mat)
            self.assertLess(torch.norm(res - actual) / actual.norm(), 0.1)

    def test_no_root_computation_when_no_cached_roots(self):
        """
        Regression test for add_low_rank speculative root computation bug.
        Verify root_decomposition is NOT called when no roots are cached.

        This catches a bug where add_low_rank would unnecessarily compute expensive
        root decompositions even when the base LinearOperator had no cached roots.
        This caused numerical instability (SVD failures) on ill-conditioned matrices.

        The fix ensures root updates only happen when BOTH:
        1. generate_roots=True (default)
        2. The base operator already has cached roots
        """
        torch.manual_seed(42)

        # Create a simple PSD matrix without any cached root decomposition
        n = 5
        A = torch.randn(n, n)
        base_matrix = A @ A.T + 0.1 * torch.eye(n)
        base_op = DenseLinearOperator(base_matrix)

        # Create a low-rank term (like LinearKernel produces)
        low_rank = torch.randn(n, 2)

        # Patch root_decomposition to track if it's called
        # Before the fix, add_low_rank would call root_decomposition even when none are cached
        # After the fix, it should NOT call root_decomposition
        with patch.object(
            DenseLinearOperator, "root_decomposition", wraps=base_op.root_decomposition
        ) as mock_root_decomp:
            result = base_op.add_low_rank(low_rank)

            # Verify root_decomposition was NOT called (the fix's behavior)
            # Before the fix, this would fail because root_decomposition was called
            # add_low_rank should NOT compute root_decomposition when no roots are cached
            self.assertEqual(mock_root_decomp.call_count, 0)

        # Verify the result is still correct (simple matrix addition)
        expected = base_matrix + low_rank @ low_rank.T
        # add_low_rank should return correct sum
        self.assertTrue(torch.allclose(result.to_dense(), expected, atol=1e-5))


class TestDenseLinearOperatorBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_op(self):
        mat = torch.randn(3, 5, 6)
        mat = mat.matmul(mat.mT)
        mat.requires_grad_(True)
        return DenseLinearOperator(mat)

    def evaluate_linear_op(self, linear_op):
        return linear_op.tensor

    def test_root_decomposition_exact(self):
        linear_op = self.create_linear_op()
        test_mat = torch.randn(*linear_op.batch_shape, linear_op.size(-1), 5)
        with linear_operator.settings.fast_computations(covar_root_decomposition=False):
            root_approx = linear_op.root_decomposition()
            res = root_approx.matmul(test_mat)
            actual = linear_op.matmul(test_mat)
            self.assertLess(torch.norm(res - actual) / actual.norm(), 0.1)


class TestDenseLinearOperatorMultiBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def create_linear_op(self):
        mat = torch.randn(2, 3, 5, 6)
        mat = mat.matmul(mat.mT)
        mat.requires_grad_(True)
        return DenseLinearOperator(mat)

    def evaluate_linear_op(self, linear_op):
        return linear_op.tensor
