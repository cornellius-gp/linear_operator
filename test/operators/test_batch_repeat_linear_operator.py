#!/usr/bin/env python3

import unittest
from unittest.mock import patch

import torch

from linear_operator import to_linear_operator
from linear_operator.operators import (
    BatchRepeatLinearOperator,
    DenseLinearOperator,
    MatmulLinearOperator,
    ToeplitzLinearOperator,
)
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase, RectangularLinearOperatorTestCase


class TestBatchRepeatLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_op(self):
        toeplitz_column = torch.tensor([4, 0.1, 0.05, 0.01, 0.0], dtype=torch.float)
        toeplitz_column.detach_()
        return BatchRepeatLinearOperator(ToeplitzLinearOperator(toeplitz_column), torch.Size((3,)))

    def evaluate_linear_op(self, linear_op):
        evaluated = linear_op.base_linear_op.to_dense()
        return evaluated.repeat(*linear_op.batch_repeat, 1, 1)


class TestBatchRepeatLinearOperatorNonSquare(RectangularLinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_op(self):
        rand_mat = torch.randn(25, 12, dtype=torch.float)
        rand_mat.detach_()
        return BatchRepeatLinearOperator(to_linear_operator(rand_mat), torch.Size((10,)))

    def evaluate_linear_op(self, linear_op):
        evaluated = linear_op.base_linear_op.to_dense()
        return evaluated.repeat(*linear_op.batch_repeat, 1, 1)


class TestBatchRepeatLinearOperatorNonSquareBatch(TestBatchRepeatLinearOperatorNonSquare):
    def create_linear_op(self) -> BatchRepeatLinearOperator:
        rand_mat = torch.randn(2, 25, 12, dtype=torch.float)
        return BatchRepeatLinearOperator(DenseLinearOperator(rand_mat), torch.Size((3,)))


class TestBatchRepeatRectangularMatmul(unittest.TestCase):
    def test_matmul_vector(self) -> None:
        for shape in [(3, 4), (2, 3, 4)]:
            matrix = torch.randn(shape, dtype=torch.double)
            operator = DenseLinearOperator(matrix).repeat(2, 1, 1)
            for transpose in (False, True):
                current = operator.mT if transpose else operator
                expected_matrix = matrix.repeat(2, 1, 1)
                if transpose:
                    expected_matrix = expected_matrix.mT
                rhs = torch.randn(current.size(-1), dtype=torch.double)
                expected = expected_matrix @ rhs
                torch.testing.assert_close(current @ rhs, expected)
                torch.testing.assert_close(current._matmul(rhs), expected)

    def test_empty_dimensions(self) -> None:
        cases = [
            ((3, 4), (0, 1, 1), (0, 4, 2)),
            ((0, 3, 4), (2, 1, 1), (0, 4, 2)),
            ((3, 0), (2, 1, 1), (2, 0, 2)),
            ((0, 4), (2, 1, 1), (2, 4, 2)),
            ((2, 3, 4), (0, 1, 1), (0, 4, 2)),
            ((2, 3, 0), (2, 1, 1), (4, 0, 2)),
        ]
        for base_shape, repeat, rhs_shape in cases:
            matrix = torch.randn(base_shape, dtype=torch.double, requires_grad=True)
            rhs = torch.randn(rhs_shape, dtype=torch.double, requires_grad=True)
            matrix_ref = matrix.detach().clone().requires_grad_(True)
            rhs_ref = rhs.detach().clone().requires_grad_(True)
            actual = DenseLinearOperator(matrix).repeat(*repeat) @ rhs
            expected = matrix_ref.repeat(*repeat) @ rhs_ref
            torch.testing.assert_close(actual, expected)
            actual_grads = torch.autograd.grad(actual.sum(), (matrix, rhs))
            expected_grads = torch.autograd.grad(expected.sum(), (matrix_ref, rhs_ref))
            for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                torch.testing.assert_close(actual_grad, expected_grad)

    def test_factorized_matmul_without_dense(self) -> None:
        for transpose in (False, True):
            left = torch.randn(2, 3, 2, dtype=torch.double, requires_grad=True)
            right = torch.randn(2, 2, 4, dtype=torch.double, requires_grad=True)
            left_ref = left.detach().clone().requires_grad_(True)
            right_ref = right.detach().clone().requires_grad_(True)
            operator = MatmulLinearOperator(DenseLinearOperator(left), DenseLinearOperator(right)).repeat(2, 1, 1)
            expected_matrix = (left_ref @ right_ref).repeat(2, 1, 1)
            if transpose:
                operator = operator.mT
                expected_matrix = expected_matrix.mT

            # Transposing storage makes the right hand side non-contiguous.
            rhs_storage = torch.randn(4, 5, operator.size(-1), dtype=torch.double, requires_grad=True)
            rhs_ref = rhs_storage.detach().clone().requires_grad_(True)
            expected = expected_matrix @ rhs_ref.mT
            with (
                patch.object(
                    BatchRepeatLinearOperator, "to_dense", side_effect=AssertionError("repeated dense matrix")
                ),
                patch.object(MatmulLinearOperator, "to_dense", side_effect=AssertionError("dense factor product")),
            ):
                actual = operator @ rhs_storage.mT
                torch.testing.assert_close(actual, expected)
                actual_grads = torch.autograd.grad(actual.square().sum(), (left, right, rhs_storage))

            expected_grads = torch.autograd.grad(expected.square().sum(), (left_ref, right_ref, rhs_ref))
            for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                torch.testing.assert_close(actual_grad, expected_grad)


class TestBatchRepeatLinearOperatorBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_op(self):
        toeplitz_column = torch.tensor([[4, 0, 0, 1], [3, 0, -0.5, -1]], dtype=torch.float)
        toeplitz_column.detach_()
        return BatchRepeatLinearOperator(ToeplitzLinearOperator(toeplitz_column), torch.Size((3,)))
        return BatchRepeatLinearOperator(ToeplitzLinearOperator(toeplitz_column), torch.Size((3,)))

    def evaluate_linear_op(self, linear_op):
        evaluated = linear_op.base_linear_op.to_dense()
        return evaluated.repeat(*linear_op.batch_repeat, 1, 1)


class TestBatchRepeatLinearOperatorMultiBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    skip_slq_tests = True

    def create_linear_op(self):
        toeplitz_column = torch.tensor(
            [[[4, 0, 0, 1], [3, 0, -0.5, -1]], [[2, 0.1, 0.01, 0.0], [3, 0, -0.1, -2]]],
            dtype=torch.float,
        )
        toeplitz_column.detach_()
        return BatchRepeatLinearOperator(ToeplitzLinearOperator(toeplitz_column), torch.Size((2, 3, 1, 4)))

    def evaluate_linear_op(self, linear_op):
        evaluated = linear_op.base_linear_op.to_dense()
        return evaluated.repeat(*linear_op.batch_repeat, 1, 1)


if __name__ == "__main__":
    unittest.main()
