#!/usr/bin/env python3

import unittest

import torch

import linear_operator.utils.toeplitz as toeplitz
from linear_operator.operators import ToeplitzLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class TestToeplitzLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    should_call_cg = False
    seed = 1

    def create_linear_op(self):
        toeplitz_column = torch.tensor([4, 0.5, 0, 1], dtype=torch.float, requires_grad=True)
        toeplitz_row = torch.tensor([4, 0.6, -0.1, -1], dtype=torch.float, requires_grad=True)
        return ToeplitzLinearOperator(toeplitz_column, toeplitz_row)

    def evaluate_linear_op(self, linear_op):
        return toeplitz.toeplitz(linear_op.column, linear_op.row)
    
    def _ensure_symmetric_grad(self, grad):
        # Hack! we don't actually want symmetric grads for this LinearOperator test case
        return grad

    # Tests that we bypass because non symmetric ToeplitzLinearOperators are not symmetric or PSD
    def test_logdet(self):
        pass

    def test_inv_quad_logdet(self):
        pass

    def test_inv_quad_logdet_no_reduce(self):
        pass

    def test_inv_quad_logdet_no_reduce_cholesky(self):
        pass

    def test_add_low_rank(self):
        pass

    def test_cat_rows(self):
        pass

    def test_cholesky(self):
        pass

    def test_diagonalization(self, symeig=False):
        pass

    def test_eigh(self):
        pass

    def test_eigvalsh(self):
        pass

    def test_root_decomposition(self, cholesky=False):
        pass

    def test_root_decomposition_cholesky(self, cholesky=False):
        pass

    def test_root_inv_decomposition(self, cholesky=False):
        pass

    def test_solve_matrix_cholesky(self):
        pass

    def test_solve_vector_with_left_cholesky(self):
        pass

    def test_sqrt_inv_matmul(self):
        pass

    def test_sqrt_inv_matmul_no_lhs(self):
        pass

    def test_svd(self):
        pass


class TestSymToeplitzLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    should_call_cg = False
    seed = 1

    def create_linear_op(self):
        toeplitz_column = torch.tensor([4, 0.5, 0, 1], dtype=torch.float, requires_grad=True)
        return ToeplitzLinearOperator(toeplitz_column)

    def evaluate_linear_op(self, linear_op):
        return toeplitz.sym_toeplitz(linear_op.column)

    # Test inv_quad_logdet and cholesky still use CG
    def test_inv_quad_logdet(self):
        self.__class__.should_call_cg = True
        super().test_inv_quad_logdet()
        self.__class__.should_call_cg = False

    def test_inv_quad_logdet_no_reduce(self):
        self.__class__.should_call_cg = True
        super().test_inv_quad_logdet_no_reduce()
        self.__class__.should_call_cg = False

    def test_root_decomposition_cholesky(self):
        self.__class__.should_call_cg = True
        super().test_root_decomposition_cholesky()
        self.__class__.should_call_cg = False


class TestSymToeplitzLinearOperatorBatch(LinearOperatorTestCase, unittest.TestCase):
    should_call_cg = False
    seed = 0

    def create_linear_op(self):
        toeplitz_column = torch.tensor([[2, -1, 0.5, 0.25], [4, 0.5, 0, 1]], dtype=torch.float, requires_grad=True)
        return ToeplitzLinearOperator(toeplitz_column)

    def evaluate_linear_op(self, linear_op):
        return torch.cat(
            [
                toeplitz.sym_toeplitz(linear_op.column[0]).unsqueeze(0),
                toeplitz.sym_toeplitz(linear_op.column[1]).unsqueeze(0),
            ]
        )

    # Test inv_quad_logdet and cholesky still use CG
    def test_inv_quad_logdet(self):
        self.__class__.should_call_cg = True
        super().test_inv_quad_logdet()
        self.__class__.should_call_cg = False

    def test_inv_quad_logdet_no_reduce(self):
        self.__class__.should_call_cg = True
        super().test_inv_quad_logdet_no_reduce()
        self.__class__.should_call_cg = False

    def test_root_decomposition_cholesky(self):
        self.__class__.should_call_cg = True
        super().test_root_decomposition_cholesky()
        self.__class__.should_call_cg = False


class TestSymToeplitzLinearOperatorMultiBatch(LinearOperatorTestCase, unittest.TestCase):
    should_call_cg = False
    seed = 0

    def create_linear_op(self):
        toeplitz_column = torch.tensor([[2, -1, 0.5, 0.25], [4, 0.5, 0, 1]], dtype=torch.float)
        toeplitz_column = toeplitz_column.repeat(3, 1, 1)
        toeplitz_column.requires_grad_(True)
        return ToeplitzLinearOperator(toeplitz_column)

    def evaluate_linear_op(self, linear_op):
        return torch.cat(
            [
                toeplitz.sym_toeplitz(linear_op.column[0, 0]).unsqueeze(0),
                toeplitz.sym_toeplitz(linear_op.column[0, 1]).unsqueeze(0),
                toeplitz.sym_toeplitz(linear_op.column[1, 0]).unsqueeze(0),
                toeplitz.sym_toeplitz(linear_op.column[1, 1]).unsqueeze(0),
                toeplitz.sym_toeplitz(linear_op.column[2, 0]).unsqueeze(0),
                toeplitz.sym_toeplitz(linear_op.column[2, 1]).unsqueeze(0),
            ]
        ).view(3, 2, 4, 4)

    # Test inv_quad_logdet and cholesky still use CG
    def test_inv_quad_logdet(self):
        self.__class__.should_call_cg = True
        super().test_inv_quad_logdet()
        self.__class__.should_call_cg = False

    def test_inv_quad_logdet_no_reduce(self):
        self.__class__.should_call_cg = True
        super().test_inv_quad_logdet_no_reduce()
        self.__class__.should_call_cg = False

    def test_root_decomposition_cholesky(self):
        self.__class__.should_call_cg = True
        super().test_root_decomposition_cholesky()
        self.__class__.should_call_cg = False


if __name__ == "__main__":
    unittest.main()
