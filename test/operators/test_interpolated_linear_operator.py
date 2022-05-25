#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import DenseLinearOperator, InterpolatedLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase, RectangularLinearOperatorTestCase


class TestInterpolatedLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 1
    should_test_sample = True

    def test_bilinear_derivative(self):
        # InterpolatedLinearOperator's representation includes int variables (the interp. indices),
        # so the default derivative doesn't apply
        pass

    def create_linear_op(self):
        left_interp_indices = torch.LongTensor([[0, 1], [2, 3], [3, 4], [4, 5]])
        left_interp_values = torch.tensor([[0.1, 0.9], [1, 2], [0.5, 1], [1, 3]], dtype=torch.float)
        left_interp_values.requires_grad = True
        right_interp_indices = torch.LongTensor([[0, 1], [2, 3], [3, 4], [4, 5]])
        right_interp_values = torch.tensor([[0.1, 0.9], [1, 2], [0.5, 1], [1, 3]], dtype=torch.float)
        right_interp_values.requires_grad = True

        base_tensor = torch.randn(6, 6)
        base_tensor = base_tensor.t().matmul(base_tensor)
        base_tensor.requires_grad = True
        base_linear_op = DenseLinearOperator(base_tensor)

        return InterpolatedLinearOperator(
            base_linear_op, left_interp_indices, left_interp_values, right_interp_indices, right_interp_values
        )

    def evaluate_linear_op(self, linear_op):
        left_matrix = torch.zeros(4, 6, dtype=linear_op.dtype)
        right_matrix = torch.zeros(4, 6, dtype=linear_op.dtype)
        left_matrix.scatter_(1, linear_op.left_interp_indices, linear_op.left_interp_values)
        right_matrix.scatter_(1, linear_op.right_interp_indices, linear_op.right_interp_values)

        base_tensor = linear_op.base_linear_op.tensor
        actual = left_matrix.matmul(base_tensor).matmul(right_matrix.t())
        return actual


class TestInterpolatedLinearOperatorBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    def test_bilinear_derivative(self):
        # InterpolatedLinearOperator's representation includes int variables (the interp. indices),
        # so the default derivative doesn't apply
        pass

    def create_linear_op(self):
        left_interp_indices = torch.LongTensor([[0, 1], [2, 3], [3, 4], [4, 5]]).repeat(5, 1, 1)
        left_interp_values = torch.tensor([[0.1, 0.9], [1, 2], [0.5, 1], [1, 3]], dtype=torch.float).repeat(5, 1, 1)
        left_interp_values.requires_grad = True
        right_interp_indices = torch.LongTensor([[0, 1], [2, 3], [3, 4], [4, 5]]).repeat(5, 1, 1)
        right_interp_values = torch.tensor([[0.1, 0.9], [1, 2], [0.5, 1], [1, 3]], dtype=torch.float).repeat(5, 1, 1)
        right_interp_values.requires_grad = True

        base_tensor = torch.randn(5, 6, 6)
        base_tensor = base_tensor.mT.matmul(base_tensor)
        base_tensor.requires_grad = True
        base_linear_op = DenseLinearOperator(base_tensor)

        return InterpolatedLinearOperator(
            base_linear_op, left_interp_indices, left_interp_values, right_interp_indices, right_interp_values
        )

    def evaluate_linear_op(self, linear_op):
        left_matrix_comps = []
        right_matrix_comps = []
        for i in range(5):
            left_matrix_comp = torch.zeros(4, 6, dtype=linear_op.dtype)
            right_matrix_comp = torch.zeros(4, 6, dtype=linear_op.dtype)
            left_matrix_comp.scatter_(1, linear_op.left_interp_indices[i], linear_op.left_interp_values[i])
            right_matrix_comp.scatter_(1, linear_op.right_interp_indices[i], linear_op.right_interp_values[i])
            left_matrix_comps.append(left_matrix_comp.unsqueeze(0))
            right_matrix_comps.append(right_matrix_comp.unsqueeze(0))
        left_matrix = torch.cat(left_matrix_comps)
        right_matrix = torch.cat(right_matrix_comps)

        base_tensor = linear_op.base_linear_op.tensor
        actual = left_matrix.matmul(base_tensor).matmul(right_matrix.mT)
        return actual


class TestInterpolatedLinearOperatorMultiBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def test_bilinear_derivative(self):
        # InterpolatedLinearOperator's representation includes int variables (the interp. indices),
        # so the default derivative doesn't apply
        pass

    def create_linear_op(self):
        left_interp_indices = torch.LongTensor([[0, 1], [2, 3], [3, 4], [4, 5]]).repeat(2, 5, 1, 1)
        left_interp_values = torch.tensor([[0.1, 0.9], [1, 2], [0.5, 1], [1, 3]], dtype=torch.float).repeat(2, 5, 1, 1)
        left_interp_values.requires_grad = True
        right_interp_indices = torch.LongTensor([[0, 1], [2, 3], [3, 4], [4, 5]]).repeat(2, 5, 1, 1)
        right_interp_values = torch.tensor([[0.1, 0.9], [1, 2], [0.5, 1], [1, 3]], dtype=torch.float).repeat(2, 5, 1, 1)
        right_interp_values.requires_grad = True

        base_tensor = torch.randn(5, 6, 6)
        base_tensor = base_tensor.mT.matmul(base_tensor)
        base_linear_op = DenseLinearOperator(base_tensor)

        return InterpolatedLinearOperator(
            base_linear_op, left_interp_indices, left_interp_values, right_interp_indices, right_interp_values
        )

    def evaluate_linear_op(self, linear_op):
        left_matrix_comps = []
        right_matrix_comps = []
        for i in range(2):
            for j in range(5):
                left_matrix_comp = torch.zeros(4, 6, dtype=linear_op.dtype)
                right_matrix_comp = torch.zeros(4, 6, dtype=linear_op.dtype)
                left_matrix_comp.scatter_(1, linear_op.left_interp_indices[i, j], linear_op.left_interp_values[i, j])
                right_matrix_comp.scatter_(1, linear_op.right_interp_indices[i, j], linear_op.right_interp_values[i, j])
                left_matrix_comps.append(left_matrix_comp.unsqueeze(0))
                right_matrix_comps.append(right_matrix_comp.unsqueeze(0))
        left_matrix = torch.cat(left_matrix_comps)
        right_matrix = torch.cat(right_matrix_comps)
        left_matrix = left_matrix.view(2, 5, 4, 6)
        right_matrix = right_matrix.view(2, 5, 4, 6)

        base_tensor = linear_op.base_linear_op.tensor
        actual = left_matrix.matmul(base_tensor).matmul(right_matrix.mT)
        return actual


def empty_method(self):
    pass


class TestInterpolatedLinearOperatorRectangular(RectangularLinearOperatorTestCase, unittest.TestCase):
    def create_linear_op(self):
        itplzt = InterpolatedLinearOperator(DenseLinearOperator(torch.rand(3, 4)))
        return itplzt

    def evaluate_linear_op(self, linear_op):
        return linear_op.base_linear_op.tensor

    # Disable tests meant for square matrices
    test_add_diag = empty_method
    test_diag = empty_method
    test_eigh = empty_method
    test_eigvalsh = empty_method
    test_inv_quad_logdet = empty_method
    test_inv_quad_logdet_no_reduce = empty_method
    test_inv_quad_logdet_no_reduce_cholesky = empty_method
    test_bilinear_derivative = empty_method
    test_root_decomposition = empty_method
    test_root_decomposition_cholesky = empty_method
    test_root_inv_decomposition = empty_method
    test_solve_matrix = empty_method
    test_solve_matrix_broadcast = empty_method
    test_solve_matrix_cholesky = empty_method
    test_solve_matrix_with_left = empty_method
    test_solve_vector = empty_method
    test_solve_vector_with_left = empty_method
    test_solve_vector_with_left_cholesky = empty_method
    test_sqrt_inv_matmul = empty_method
    test_sqrt_inv_matmul_no_lhs = empty_method
    test_svd = empty_method


if __name__ == "__main__":
    unittest.main()
