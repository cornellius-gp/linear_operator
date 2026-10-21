import unittest

import torch

from linear_operator.operators import (
    BlockDiagLinearOperator,
    CatLinearOperator,
    DenseLinearOperator,
    DiagLinearOperator,
    MatmulLinearOperator,
)


class TestCatHigherDerivatives(unittest.TestCase):
    def test_matmul_mixed_higher_derivatives(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for dim in (-1, -2, -3, -4):
                for rhs_batch in (False, True):
                    with self.subTest(device=device, dim=dim, rhs_batch=rhs_batch):
                        torch.manual_seed(12)
                        sizes = [2, 3, 2, 3]
                        first = sizes.copy()
                        second = sizes.copy()
                        first[dim], second[dim] = 1, sizes[dim] - 1
                        left = torch.randn(*first, device=device, dtype=torch.double, requires_grad=True)
                        right = torch.randn(*second, device=device, dtype=torch.double, requires_grad=True)
                        dense = torch.cat((left, right), dim=dim)
                        shape = (4, *dense.shape[:-2], dense.shape[-1], 2) if rhs_batch else (dense.shape[-1], 2)
                        rhs = torch.randn(*shape, device=device, dtype=torch.double, requires_grad=True)
                        values = (left, right, rhs)
                        operator = CatLinearOperator(DenseLinearOperator(left), DenseLinearOperator(right), dim=dim)
                        actual = operator @ rhs
                        expected = dense @ rhs
                        torch.testing.assert_close(actual, expected)
                        actual_loss = actual.sin().sum() + 0.2 * actual.square().sum()
                        expected_loss = expected.sin().sum() + 0.2 * expected.square().sum()
                        for order in range(3):
                            actual_grad = torch.autograd.grad(actual_loss, values, create_graph=order < 2)
                            expected_grad = torch.autograd.grad(expected_loss, values, create_graph=order < 2)
                            for observed, reference in zip(actual_grad, expected_grad):
                                torch.testing.assert_close(observed, reference, rtol=2e-7, atol=2e-8)
                            if order < 2:
                                actual_loss = sum(g.square().sum() for g in actual_grad)
                                expected_loss = sum(g.square().sum() for g in expected_grad)

    def test_independent_covariance_dense_hessian(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                first = torch.tensor([[[1.0, 0.3], [0.3, 2.0]]], device=device, dtype=torch.double, requires_grad=True)
                second = torch.tensor([[[0.6, 0.2], [0.2, 1.0]]], device=device, dtype=torch.double, requires_grad=True)
                blocks = CatLinearOperator(
                    DenseLinearOperator(first), DenseLinearOperator(second), dim=0, output_device=first.device
                )
                covariance = BlockDiagLinearOperator(blocks).to_dense()
                gradients = torch.autograd.grad(covariance.square().sum(), (first, second), create_graph=True)
                for gradient, value in zip(gradients, (first, second)):
                    torch.testing.assert_close(gradient, 2 * value)
                hessian = torch.autograd.grad(sum(g.square().sum() for g in gradients), (first, second))
                for observed, value in zip(hessian, (first, second)):
                    torch.testing.assert_close(observed, 8 * value)

    def test_gradcheck_and_gradgradcheck(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for dim in (-1, -2, -3):
                for vector in (False, True):
                    if vector and dim == -3:
                        continue
                    with self.subTest(device=device, dim=dim, vector=vector):
                        torch.manual_seed(16)
                        first = torch.randn(
                            (2, 2) if vector else (1, 2, 2),
                            device=device,
                            dtype=torch.double,
                            requires_grad=True,
                        )
                        second = torch.randn_like(first, requires_grad=True)
                        joined = torch.cat((first, second), dim=dim)
                        rhs = torch.randn(
                            (joined.shape[-1],) if vector else (joined.shape[-1], 2),
                            device=device,
                            dtype=torch.double,
                            requires_grad=True,
                        )

                        def evaluate(left, right, vectors):
                            return (
                                CatLinearOperator(DenseLinearOperator(left), DenseLinearOperator(right), dim=dim)
                                @ vectors
                            )

                        self.assertTrue(torch.autograd.gradcheck(evaluate, (first, second, rhs), fast_mode=True))
                        self.assertTrue(torch.autograd.gradgradcheck(evaluate, (first, second, rhs), fast_mode=True))

    def test_nested_children_and_cross_device_gradients(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for first_device in devices:
            for second_device in devices:
                for output_device in devices:
                    with self.subTest(first=first_device, second=second_device, output=output_device):
                        torch.manual_seed(18)
                        left = torch.randn(2, 2, device=first_device, dtype=torch.double, requires_grad=True)
                        right = torch.randn_like(left, requires_grad=True)
                        diagonal = torch.randn(2, device=second_device, dtype=torch.double, requires_grad=True)
                        rhs = torch.randn(2, 3, device=output_device, dtype=torch.double, requires_grad=True)
                        operator = CatLinearOperator(
                            MatmulLinearOperator(left, right),
                            DiagLinearOperator(diagonal),
                            dim=-2,
                            output_device=torch.device(output_device),
                        )
                        actual = operator @ rhs
                        dense = torch.cat(((left @ right).to(output_device), diagonal.diag().to(output_device)), dim=-2)
                        expected = dense @ rhs
                        params = (left, right, diagonal, rhs)
                        actual_loss, expected_loss = actual.square().sum(), expected.square().sum()
                        for order in range(3):
                            observed = torch.autograd.grad(
                                actual_loss, params, create_graph=order < 2, retain_graph=True
                            )
                            reference = torch.autograd.grad(
                                expected_loss, params, create_graph=order < 2, retain_graph=True
                            )
                            for a, b in zip(observed, reference):
                                torch.testing.assert_close(a, b, rtol=2e-7, atol=2e-8)
                            if order < 2:
                                actual_loss = sum(g.square().sum().to(output_device) for g in observed)
                                expected_loss = sum(g.square().sum().to(output_device) for g in reference)
