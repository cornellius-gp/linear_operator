#!/usr/bin/env python3

import unittest

import torch

from linear_operator import solve
from linear_operator.operators import DenseLinearOperator, DiagLinearOperator, TriangularLinearOperator
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

        inverse = torch.linalg.inv(linear_op).to_dense()
        inverse_actual = evaluated.inverse()
        self.assertAllClose(inverse, inverse_actual)

        # Verify deprecated torch.inverse also dispatches correctly
        inverse_deprecated = torch.inverse(linear_op).to_dense()
        self.assertAllClose(inverse, inverse_deprecated)

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


class TestStructuredTriangularSolve(unittest.TestCase):
    def test_left_tensor_with_added_diagonal(self):
        for dtype in (torch.float32, torch.float64):
            for upper in (False, True):
                for vector in (False, True):
                    for rows in (2, 3):
                        with self.subTest(dtype=dtype, upper=upper, vector=vector, rows=rows):
                            diagonal = torch.tensor([2.0, 3.0, 4.0], dtype=dtype)
                            noise = torch.tensor([0.2, 0.3, 0.5], dtype=dtype)
                            operator = TriangularLinearOperator(diagonal.diag(), upper=upper)
                            operator = operator + DiagLinearOperator(noise)
                            rhs = torch.arange(1, 7, dtype=dtype).reshape(3, 2)
                            rhs = rhs[:, 0] if vector else rhs
                            lhs = torch.arange(1, 1 + rows * 3, dtype=dtype).reshape(rows, 3) / 4
                            expected = lhs @ torch.linalg.solve((diagonal + noise).diag(), rhs)
                            torch.testing.assert_close(solve(operator, rhs, lhs), expected)

    def test_left_tensor_with_repeated_factor(self):
        for dtype in (torch.float32, torch.float64):
            for upper in (False, True):
                for vector in (False, True):
                    for rows in (2, 3):
                        with self.subTest(dtype=dtype, upper=upper, vector=vector, rows=rows):
                            factor = torch.tensor([[2.0, 0.0, 0.0], [0.3, 3.0, 0.0], [0.1, -0.2, 4.0]], dtype=dtype)
                            factor = factor.mT if upper else factor
                            operator = TriangularLinearOperator(factor, upper=upper)
                            operator = TriangularLinearOperator(operator.repeat(2, 1, 1), upper=upper)
                            rhs = torch.arange(1, 7, dtype=dtype).reshape(3, 2)
                            rhs = rhs[:, 0] if vector else rhs
                            lhs = torch.arange(1, 1 + rows * 3, dtype=dtype).reshape(rows, 3) / 4
                            expected = lhs @ torch.linalg.solve_triangular(
                                factor, rhs[:, None] if vector else rhs, upper=upper
                            )
                            expected = expected.squeeze(-1) if vector else expected
                            expected = expected.expand(2, *expected.shape)
                            torch.testing.assert_close(solve(operator, rhs, lhs), expected)

    def test_broadcast_left_tensor_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for upper in (False, True):
                with self.subTest(dtype=dtype, upper=upper):
                    diagonal = torch.tensor([[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]], dtype=dtype, requires_grad=True)
                    noise = torch.tensor([0.2, 0.3, 0.5], dtype=dtype, requires_grad=True)
                    rhs = torch.arange(1, 7, dtype=dtype).reshape(1, 3, 2).requires_grad_()
                    lhs = (torch.arange(1, 19, dtype=dtype).reshape(2, 3, 3) / 4).requires_grad_()
                    operator = TriangularLinearOperator(diagonal.diag_embed(), upper=upper)
                    operator = operator + DiagLinearOperator(noise)
                    actual = solve(operator, rhs, lhs)
                    expected = lhs @ torch.linalg.solve((diagonal + noise).diag_embed(), rhs)
                    torch.testing.assert_close(actual, expected)
                    inputs = (diagonal, noise, rhs, lhs)
                    actual_grads = torch.autograd.grad(actual.square().sum(), inputs)
                    expected_grads = torch.autograd.grad(expected.square().sum(), inputs)
                    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                        torch.testing.assert_close(actual_grad, expected_grad)


if __name__ == "__main__":
    unittest.main()
