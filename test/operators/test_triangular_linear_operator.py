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

    def test_matmul_gradgrad(self):
        for upper in (False, True):
            for vector in (False, True):
                with self.subTest(upper=upper, vector=vector):
                    raw = torch.tensor([[1.2, 0.1], [0.3, 0.9]], dtype=torch.double, requires_grad=True)
                    rhs = torch.tensor([[0.7, 0.2], [-0.4, 0.6]], dtype=torch.double)
                    rhs = (rhs[:, 0] if vector else rhs).requires_grad_()

                    def matmul(raw, rhs):
                        triangle = raw.triu() if upper else raw.tril()
                        return TriangularLinearOperator(triangle, upper=upper).matmul(rhs).square()

                    self.assertTrue(torch.autograd.gradcheck(matmul, (raw, rhs)))
                    self.assertTrue(torch.autograd.gradgradcheck(matmul, (raw, rhs)))

    def test_cholesky_matmul_second_derivative(self):
        raw = torch.tensor([[1.2, 0.1], [0.3, 0.9]], dtype=torch.double, requires_grad=True)
        rhs = torch.tensor([0.7, -0.4], dtype=torch.double)
        gradients = []
        for wrapped in (False, True):
            covariance = raw @ raw.mT + torch.eye(2, dtype=torch.double)
            root = torch.linalg.cholesky(covariance)
            sample = TriangularLinearOperator(root).matmul(rhs) if wrapped else root @ rhs
            first = torch.autograd.grad(sample.square().sum(), raw, create_graph=True)[0]
            second = torch.autograd.grad(first.sum(), raw)[0]
            gradients.append((first, second))
        for actual, expected in zip(*gradients):
            self.assertAllClose(actual, expected)

    def test_matrix_hessian_with_constant_rhs(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            with self.subTest(device=device):
                root = torch.tensor(
                    [[1.2, 0.0], [0.3, 0.9]],
                    dtype=torch.double,
                    device=device,
                    requires_grad=True,
                )
                rhs = root.new_tensor([0.7, -0.4])
                wrapped = TriangularLinearOperator(root).matmul(rhs)
                first = torch.autograd.grad(wrapped.square().sum(), root, create_graph=True)[0]
                expected = torch.autograd.grad((root @ rhs).square().sum(), root, create_graph=True)[0]
                self.assertAllClose(first, expected)
                self.assertTrue(first.requires_grad)
                self.assertAllClose(
                    torch.autograd.grad(first.sum(), root)[0],
                    torch.autograd.grad(expected.sum(), root)[0],
                )

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


if __name__ == "__main__":
    unittest.main()
