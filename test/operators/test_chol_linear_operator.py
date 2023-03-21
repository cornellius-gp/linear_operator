#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import CholLinearOperator, TriangularLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class TestCholLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True
    should_call_cg = False
    should_call_lanczos = False
    should_call_lanczos_diagonalization = True

    def create_linear_op(self):
        chol = torch.tensor(
            [[3, 0, 0, 0, 0], [-1, 2, 0, 0, 0], [1, 4, 1, 0, 0], [0, 2, 3, 2, 0], [-4, -2, 1, 3, 4]],
            dtype=torch.float,
            requires_grad=True,
        )
        return CholLinearOperator(TriangularLinearOperator(chol))

    def evaluate_linear_op(self, linear_op):
        chol = linear_op.root.to_dense()
        return chol.matmul(chol.mT)

    def test_inverse(self):
        linear_op = self.create_linear_op()
        linear_op_copy = linear_op.detach().clone()
        linear_op.requires_grad_(True)
        linear_op_copy.requires_grad_(True)
        evaluated = self.evaluate_linear_op(linear_op_copy)

        inverse = torch.inverse(linear_op).to_dense()
        inverse_actual = evaluated.inverse()
        self.assertAllClose(inverse, inverse_actual)

        # Backwards
        inverse.sum().backward()
        inverse_actual.sum().backward()

        # Check grads
        for arg, arg_copy in zip(linear_op.representation(), linear_op_copy.representation()):
            if arg_copy.requires_grad and arg_copy.is_leaf and arg_copy.grad is not None:
                self.assertAllClose(arg.grad.tril(), arg_copy.grad.tril())


class TestCholLinearOperatorBatch(TestCholLinearOperator):
    seed = 0

    def create_linear_op(self):
        chol = torch.tensor(
            [
                [[3, 0, 0, 0, 0], [-1, 2, 0, 0, 0], [1, 4, 1, 0, 0], [0, 2, 3, 2, 0], [-4, -2, 1, 3, 4]],
                [[2, 0, 0, 0, 0], [3, 1, 0, 0, 0], [-2, 3, 2, 0, 0], [-2, 1, -1, 3, 0], [-4, -4, 5, 2, 3]],
            ],
            dtype=torch.float,
        )
        chol.add_(torch.eye(5).unsqueeze(0))
        chol.requires_grad_(True)
        return CholLinearOperator(TriangularLinearOperator(chol))


class TestCholLinearOperatorMultiBatch(TestCholLinearOperator):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def create_linear_op(self):
        chol = torch.tensor(
            [
                [[3, 0, 0, 0, 0], [-1, 2, 0, 0, 0], [1, 4, 1, 0, 0], [0, 2, 3, 2, 0], [-4, -2, 1, 3, 4]],
                [[2, 0, 0, 0, 0], [3, 1, 0, 0, 0], [-2, 3, 2, 0, 0], [-2, 1, -1, 3, 0], [-4, -4, 5, 2, 3]],
            ],
            dtype=torch.float,
        )
        chol = chol.repeat(3, 1, 1, 1)
        chol[1].mul_(2)
        chol[2].mul_(0.5)
        chol.add_(torch.eye(5).unsqueeze_(0).unsqueeze_(0))
        chol.requires_grad_(True)
        return CholLinearOperator(TriangularLinearOperator(chol))


if __name__ == "__main__":
    unittest.main()
