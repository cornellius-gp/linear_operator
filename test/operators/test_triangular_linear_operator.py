#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import DenseLinearOperator, TriangularLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


# TODO: create test suite for square, but non-symmetric/PSD linear operators
class TestTriangularLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = False
    should_call_cg = False
    should_call_lanczos = False

    def _ensure_symmetric_grad(self, grad):
        # Hack! we don't actually want symmetric grads for this LinearOperator test case
        # We actually want triangular gradients
        return grad.tril()

    def create_linear_op(self):
        tensor = torch.randn(5, 5).tril()

        # Make a positive diagonal
        diag = tensor.diagonal(dim1=-1, dim2=-2).diag_embed()
        tensor = tensor - diag + diag.abs()

        tensor.requires_grad_(True)
        return TriangularLinearOperator(DenseLinearOperator(tensor))

    def evaluate_linear_op(self, linear_op):
        tensor = linear_op._tensor.tensor
        return tensor

    def test_inverse(self):
        linear_op = self.create_linear_op()
        linear_op_copy = linear_op.detach().clone()
        linear_op.requires_grad_(True)
        linear_op_copy.requires_grad_(True)
        evaluated = self.evaluate_linear_op(linear_op_copy)
        evaluated.register_hook(self._ensure_symmetric_grad)

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

    def test_solve_triangular(self):
        linear_op = self.create_linear_op()
        is_upper = linear_op.upper
        rhs = torch.randn(linear_op.size(-1), 1)
        res = torch.linalg.solve_triangular(linear_op, rhs, upper=is_upper)
        res_actual = torch.linalg.solve_triangular(linear_op.to_dense(), rhs, upper=is_upper)
        self.assertAllClose(res, res_actual)
        with self.assertRaisesRegex(RuntimeError, "Incompatible argument"):
            torch.linalg.solve_triangular(linear_op, rhs, upper=not is_upper)

    # Tests that we bypass because TriangularLinearOperators are not symmetric or PSD

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

    def test_sqrt_inv_matmul(self):
        pass

    def test_sqrt_inv_matmul_no_lhs(self):
        pass

    def test_svd(self):
        pass


class TestUpperTriangularLinearOperator(TestTriangularLinearOperator):
    def _ensure_symmetric_grad(self, grad):
        # Hack! we don't actually want symmetric grads for this LinearOperator test case
        # We actually want triangular gradients
        return grad.triu()

    def create_linear_op(self):
        tensor = torch.randn(5, 5).triu()

        # Make a positive diagonal
        diag = tensor.diagonal(dim1=-1, dim2=-2).diag_embed()
        tensor = tensor - diag + diag.abs()

        tensor.requires_grad_(True)
        res = TriangularLinearOperator(DenseLinearOperator(tensor), upper=True)
        return res

    def evaluate_linear_op(self, linear_op):
        tensor = linear_op._tensor.tensor
        return tensor


if __name__ == "__main__":
    unittest.main()
