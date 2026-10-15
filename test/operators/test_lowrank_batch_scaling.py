#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import DiagLinearOperator, LowRankRootAddedDiagLinearOperator, LowRankRootLinearOperator


class TestLowRankBatchScaling(unittest.TestCase):
    def test_batch_scale_values_and_gradients(self) -> None:
        torch.manual_seed(6)
        for batch in ((3,), (2, 3)):
            with self.subTest(batch=batch):
                root = torch.randn(*batch, 8, 3, dtype=torch.double, requires_grad=True)
                diagonal = torch.ones(*batch, 8, dtype=torch.double, requires_grad=True)
                scales = torch.linspace(0.5, 2.0, 3, dtype=torch.double, requires_grad=True)
                copies = [value.detach().clone().requires_grad_() for value in (root, diagonal, scales)]
                operator = LowRankRootLinearOperator(root) + DiagLinearOperator(diagonal)
                scaled = operator * scales[:, None, None]
                self.assertIsInstance(scaled, LowRankRootAddedDiagLinearOperator)
                actual = scaled.to_dense()
                dense = copies[0] @ copies[0].mT + torch.diag_embed(copies[1])
                expected = dense * copies[2][:, None, None]
                torch.testing.assert_close(actual, expected)
                actual_grads = torch.autograd.grad(actual.square().sum(), (root, diagonal, scales))
                expected_grads = torch.autograd.grad(expected.square().sum(), copies)
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual_grad, expected_grad)

    def test_batch_with_zero_factor(self) -> None:
        root = torch.ones(3, 8, 2, dtype=torch.double)
        operator = LowRankRootLinearOperator(root).add_diagonal(torch.ones(3, 8, dtype=torch.double))
        scales = torch.tensor([0.0, 0.5, 2.0], dtype=torch.double)[:, None, None]
        torch.testing.assert_close((operator * scales).to_dense(), operator.to_dense() * scales)
