#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import DiagLinearOperator, LowRankRootLinearOperator


class TestLowRankVectorQuadratic(unittest.TestCase):
    def test_vector_values_and_gradients(self) -> None:
        torch.manual_seed(14)
        for reduce in (False, True):
            for logdet in (False, True):
                with self.subTest(reduce=reduce, logdet=logdet):
                    root = torch.randn(8, 3, dtype=torch.double, requires_grad=True)
                    diagonal = torch.linspace(0.5, 1.5, 8, dtype=torch.double, requires_grad=True)
                    rhs = torch.randn(8, dtype=torch.double, requires_grad=True)
                    copies = [value.detach().clone().requires_grad_() for value in (root, diagonal, rhs)]
                    dense = copies[0] @ copies[0].mT + torch.diag(copies[1])
                    operator = LowRankRootLinearOperator(root) + DiagLinearOperator(diagonal)
                    actual_q, actual_l = operator.inv_quad_logdet(rhs, logdet=logdet, reduce_inv_quad=reduce)
                    expected_q = (copies[2] * torch.linalg.solve(dense, copies[2])).sum()
                    if not reduce:
                        expected_q = expected_q.unsqueeze(0)
                    torch.testing.assert_close(actual_q, expected_q)
                    actual_loss, expected_loss = actual_q.sum(), expected_q.sum()
                    if logdet:
                        expected_l = torch.linalg.slogdet(dense)[1]
                        torch.testing.assert_close(actual_l, expected_l)
                        actual_loss = actual_loss + actual_l
                        expected_loss = expected_loss + expected_l
                    actual_grads = torch.autograd.grad(actual_loss, (root, diagonal, rhs))
                    expected_grads = torch.autograd.grad(expected_loss, copies)
                    for actual, expected in zip(actual_grads, expected_grads):
                        torch.testing.assert_close(actual, expected)

    def test_vector_matches_single_column(self) -> None:
        torch.manual_seed(1)
        root = torch.randn(8, 3, dtype=torch.double)
        operator = LowRankRootLinearOperator(root).add_diagonal(torch.ones(8, dtype=torch.double))
        rhs = torch.randn(8, dtype=torch.double)
        vector = operator.inv_quad_logdet(rhs, logdet=True, reduce_inv_quad=False)
        column = operator.inv_quad_logdet(rhs.unsqueeze(-1), logdet=True, reduce_inv_quad=False)
        for actual, expected in zip(vector, column):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_wrong_vector_length(self) -> None:
        root = torch.ones(8, 2, dtype=torch.double)
        operator = LowRankRootLinearOperator(root).add_diagonal(torch.ones(8, dtype=torch.double))
        with self.assertRaisesRegex(RuntimeError, "cannot be multiplied"):
            operator.inv_quad_logdet(torch.ones(7, dtype=torch.double))
