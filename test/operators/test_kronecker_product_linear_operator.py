#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import DenseLinearOperator, KroneckerProductLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase, RectangularLinearOperatorTestCase


def kron(a, b):
    res = []
    for i in range(a.size(-2)):
        row_res = []
        for j in range(a.size(-1)):
            row_res.append(b * a[..., i, j].unsqueeze(-1).unsqueeze(-2))
        res.append(torch.cat(row_res, -1))
    return torch.cat(res, -2)


class TestKroneckerProductLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    should_call_lanczos = True
    should_call_lanczos_diagonalization = False

    def create_linear_op(self):
        a = torch.tensor([[4, 0, 2], [0, 3, -1], [2, -1, 3]], dtype=torch.float)
        b = torch.tensor([[2, 1], [1, 2]], dtype=torch.float)
        c = torch.tensor(
            [[4, 0.5, 1, 0], [0.5, 4, -1, 0], [1, -1, 3, 0], [0, 0, 0, 4]],
            dtype=torch.float,
        )
        a.requires_grad_(True)
        b.requires_grad_(True)
        c.requires_grad_(True)
        kp_linear_op = KroneckerProductLinearOperator(
            DenseLinearOperator(a), DenseLinearOperator(b), DenseLinearOperator(c)
        )
        return kp_linear_op

    def evaluate_linear_op(self, linear_op):
        res = kron(linear_op.linear_ops[0].tensor, linear_op.linear_ops[1].tensor)
        res = kron(res, linear_op.linear_ops[2].tensor)
        return res


class TestKroneckerProductLinearOperatorBatch(TestKroneckerProductLinearOperator):
    seed = 0
    should_call_lanczos = True
    should_call_lanczos_diagonalization = False

    def create_linear_op(self):
        a = torch.tensor([[4, 0, 2], [0, 3, -1], [2, -1, 3]], dtype=torch.float).repeat(3, 1, 1)
        b = torch.tensor([[2, 1], [1, 2]], dtype=torch.float).repeat(3, 1, 1)
        c = torch.tensor(
            [[4, 0.1, 1, 0], [0.1, 4, -1, 0], [1, -1, 3, 0], [0, 0, 0, 4]],
            dtype=torch.float,
        ).repeat(3, 1, 1)
        a.requires_grad_(True)
        b.requires_grad_(True)
        c.requires_grad_(True)
        kp_linear_op = KroneckerProductLinearOperator(
            DenseLinearOperator(a), DenseLinearOperator(b), DenseLinearOperator(c)
        )
        return kp_linear_op


class TestKroneckerProductLinearOperatorRectangular(RectangularLinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_op(self):
        a = torch.randn(2, 3, requires_grad=True)
        b = torch.randn(5, 2, requires_grad=True)
        c = torch.randn(6, 4, requires_grad=True)
        kp_linear_op = KroneckerProductLinearOperator(
            DenseLinearOperator(a), DenseLinearOperator(b), DenseLinearOperator(c)
        )
        return kp_linear_op

    def evaluate_linear_op(self, linear_op):
        res = kron(linear_op.linear_ops[0].tensor, linear_op.linear_ops[1].tensor)
        res = kron(res, linear_op.linear_ops[2].tensor)
        return res


class TestKroneckerProductLinearOperatorRectangularMultiBatch(TestKroneckerProductLinearOperatorRectangular):
    seed = 0

    def create_linear_op(self):
        a = torch.randn(3, 4, 2, 3, requires_grad=True)
        b = torch.randn(3, 4, 5, 2, requires_grad=True)
        c = torch.randn(3, 4, 6, 4, requires_grad=True)
        kp_linear_op = KroneckerProductLinearOperator(
            DenseLinearOperator(a), DenseLinearOperator(b), DenseLinearOperator(c)
        )
        return kp_linear_op


class TestKroneckerProductLogdetScale(unittest.TestCase):
    def test_logdet_and_gradients_across_scales(self):
        for dtype in (torch.float32, torch.float64):
            for scale in (1.0, 1e-4, 1e-8):
                with self.subTest(dtype=dtype, scale=scale):
                    a = (scale * torch.tensor([[2.0, 0.3], [0.3, 1.0]], dtype=dtype)).requires_grad_()
                    b = scale * torch.tensor([[1.0, 0.1, 0.0], [0.1, 2.0, 0.2], [0.0, 0.2, 1.0]], dtype=dtype)
                    b.requires_grad_()
                    operator = KroneckerProductLinearOperator(a, b)
                    actual = operator.logdet()
                    expected = torch.linalg.slogdet(torch.kron(a, b)).logabsdet
                    torch.testing.assert_close(actual, expected)
                    actual_grads = torch.autograd.grad(actual, (a, b))
                    # d logdet(A kron B) / dA = dim(B) * A^{-T}.
                    rtol = 5e-6 if dtype == torch.float32 else 1e-7
                    torch.testing.assert_close(actual_grads[0], 3 * torch.linalg.inv(a).mT, rtol=rtol, atol=1e-7)
                    torch.testing.assert_close(actual_grads[1], 2 * torch.linalg.inv(b).mT, rtol=rtol, atol=1e-7)

    def test_inv_quad_logdet_preserves_gaussian_likelihood(self):
        for dtype in (torch.float32, torch.float64):
            for reduce_inv_quad in (False, True):
                with self.subTest(dtype=dtype, reduce_inv_quad=reduce_inv_quad):
                    scale = torch.tensor(1e-4, dtype=dtype, requires_grad=True)
                    a = scale * torch.tensor([[2.0, 0.3], [0.3, 1.0]], dtype=dtype)
                    b = 1e-4 * torch.tensor([[1.0, 0.1, 0.0], [0.1, 2.0, 0.2], [0.0, 0.2, 1.0]], dtype=dtype)
                    rhs = 1e-4 * torch.arange(12, dtype=dtype).reshape(6, 2)
                    operator = KroneckerProductLinearOperator(a, b)
                    inv_quad, logdet = operator.inv_quad_logdet(rhs, logdet=True, reduce_inv_quad=reduce_inv_quad)
                    dense = torch.kron(a, b)
                    expected_quad = (rhs * torch.linalg.solve(dense, rhs)).sum(-2)
                    if reduce_inv_quad:
                        expected_quad = expected_quad.sum(-1)
                    expected_logdet = torch.linalg.slogdet(dense).logabsdet
                    torch.testing.assert_close(inv_quad, expected_quad)
                    torch.testing.assert_close(logdet, expected_logdet)
                    actual_loss = (inv_quad + logdet).sum()
                    expected_loss = (expected_quad + expected_logdet).sum()
                    torch.testing.assert_close(
                        torch.autograd.grad(actual_loss, scale, retain_graph=True)[0],
                        torch.autograd.grad(expected_loss, scale)[0],
                    )

    def test_logdet_broadcast_batches_and_factor_multiplicities(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                a = torch.tensor([[2.0, 0.3], [0.3, 1.0]], dtype=dtype).expand(2, 1, 2, 2)
                a = (a * torch.tensor([1.0, 1e-5], dtype=dtype).view(2, 1, 1, 1)).requires_grad_()
                b = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=dtype)).expand(1, 3, 3, 3)
                b = (b * torch.tensor([1.0, 1e-4, 1e-8], dtype=dtype).view(1, 3, 1, 1)).requires_grad_()
                c = torch.tensor([[1.0, 0.2], [0.2, 1.5]], dtype=dtype, requires_grad=True)
                actual = KroneckerProductLinearOperator(a, b, c).logdet()
                dense = kron(kron(a, b), c)
                expected = torch.linalg.slogdet(dense).logabsdet
                self.assertEqual(actual.shape, torch.Size([2, 3]))
                torch.testing.assert_close(actual, expected)
                actual_grads = torch.autograd.grad(actual.sum(), (a, b, c), retain_graph=True)
                expected_grads = torch.autograd.grad(expected.sum(), (a, b, c))
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual_grad, expected_grad)

    def test_logdet_avoids_product_underflow_and_overflow(self):
        for dtype, exponent in ((torch.float32, 30.0), (torch.float64, 200.0)):
            for sign in (-1, 1):
                with self.subTest(dtype=dtype, sign=sign):
                    log_scale = torch.tensor(sign * exponent, dtype=dtype, requires_grad=True)
                    scale = 10.0**log_scale
                    a = torch.eye(2, dtype=dtype) * scale
                    b = torch.eye(3, dtype=dtype) * scale
                    actual = KroneckerProductLinearOperator(a, b).logdet()
                    expected = 12 * log_scale * torch.tensor(10.0, dtype=dtype).log()
                    self.assertTrue(actual.isfinite())
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(
                        torch.autograd.grad(actual, log_scale)[0],
                        torch.tensor(12.0, dtype=dtype) * torch.tensor(10.0, dtype=dtype).log(),
                    )

    def test_logdet_zero_eigenvalue(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                a = torch.diag(torch.tensor([1.0, 0.0], dtype=dtype))
                b = torch.eye(3, dtype=dtype)
                actual = KroneckerProductLinearOperator(a, b).logdet()
                self.assertEqual(actual.item(), -float("inf"))

    def test_logdet_first_and_second_derivatives(self):
        a = 1e-4 * torch.tensor([[2.0, 0.3], [0.3, 1.0]], dtype=torch.float64)
        b = 1e-4 * torch.tensor([[1.0, 0.1, 0.0], [0.1, 2.0, 0.2], [0.0, 0.2, 1.0]], dtype=torch.float64)
        log_scale = torch.tensor(-1.0, dtype=torch.float64, requires_grad=True)

        def logdet(log_scale):
            return KroneckerProductLinearOperator(log_scale.exp() * a, b).logdet()

        self.assertTrue(torch.autograd.gradcheck(logdet, (log_scale,)))
        self.assertTrue(torch.autograd.gradgradcheck(logdet, (log_scale,)))
        torch.testing.assert_close(
            torch.autograd.grad(logdet(log_scale), log_scale)[0], torch.tensor(6.0, dtype=torch.float64)
        )


if __name__ == "__main__":
    unittest.main()
