#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import ToeplitzLinearOperator, to_linear_operator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class TestSumLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_op(self):
        c1 = torch.tensor([5, 1, 2, 0], dtype=torch.float, requires_grad=True)
        t1 = ToeplitzLinearOperator(c1)
        c2 = torch.tensor([6, 0, 1, -1], dtype=torch.float, requires_grad=True)
        t2 = ToeplitzLinearOperator(c2)
        return t1 + t2

    def evaluate_linear_op(self, linear_op):
        tensors = [lt.to_dense() for lt in linear_op.linear_ops]
        return sum(tensors)


class TestSumLinearOperatorBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_op(self):
        c1 = torch.tensor([[2, 0.5, 0, 0], [5, 1, 2, 0]], dtype=torch.float, requires_grad=True)
        t1 = ToeplitzLinearOperator(c1)
        c2 = torch.tensor([[2, 0.5, 0, 0], [6, 0, 1, -1]], dtype=torch.float, requires_grad=True)
        t2 = ToeplitzLinearOperator(c2)
        return t1 + t2

    def evaluate_linear_op(self, linear_op):
        tensors = [lt.to_dense() for lt in linear_op.linear_ops]
        return sum(tensors)


class TestSumLinearOperatorMultiBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    skip_slq_tests = True

    def create_linear_op(self):
        c1 = torch.tensor(
            [[[2, 0.5, 0, 0], [5, 1, 2, 0]], [[2, 0.5, 0, 0], [5, 1, 2, 0]]],
            dtype=torch.float,
            requires_grad=True,
        )
        t1 = ToeplitzLinearOperator(c1)
        c2 = torch.tensor(
            [[[2, 0.5, 0, 0], [5, 1, 2, 0]], [[2, 0.5, 0, 0], [6, 0, 1, -1]]],
            dtype=torch.float,
            requires_grad=True,
        )
        t2 = ToeplitzLinearOperator(c2)
        return t1 + t2

    def evaluate_linear_op(self, linear_op):
        tensors = [lt.to_dense() for lt in linear_op.linear_ops]
        return sum(tensors)


class TestSumLinearOperatorBroadcasting(unittest.TestCase):
    def test_broadcast_same_shape(self):
        test1 = to_linear_operator(torch.randn(30, 30))

        test2 = torch.randn(30, 30)
        res = test1 + test2
        final_res = res + test2

        torch_res = res.to_dense() + test2

        self.assertEqual(final_res.shape, torch_res.shape)
        self.assertEqual((final_res.to_dense() - torch_res).sum(), 0.0)

    def test_broadcast_tensor_shape(self):
        test1 = to_linear_operator(torch.randn(30, 30))

        test2 = torch.randn(30, 1)
        res = test1 + test2
        final_res = res + test2

        torch_res = res.to_dense() + test2

        self.assertEqual(final_res.shape, torch_res.shape)
        self.assertEqual((final_res.to_dense() - torch_res).sum(), 0.0)

    def test_broadcast_lazy_shape(self):
        test1 = to_linear_operator(torch.randn(30, 1))

        test2 = torch.randn(30, 30)
        res = test1 + test2
        final_res = res + test2

        torch_res = res.to_dense() + test2

        self.assertEqual(final_res.shape, torch_res.shape)
        self.assertEqual((final_res.to_dense() - torch_res).sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
