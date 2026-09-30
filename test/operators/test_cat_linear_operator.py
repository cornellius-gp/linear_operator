#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import CatLinearOperator, DenseLinearOperator, DiagLinearOperator, IdentityLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class TestCatLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 1

    def create_linear_op(self):
        root = torch.randn(6, 7)
        self.psd_mat = root.matmul(root.t())

        slice1_mat = self.psd_mat[:2, :].requires_grad_()
        slice2_mat = self.psd_mat[2:4, :].requires_grad_()
        slice3_mat = self.psd_mat[4:6, :].requires_grad_()

        slice1 = DenseLinearOperator(slice1_mat)
        slice2 = DenseLinearOperator(slice2_mat)
        slice3 = DenseLinearOperator(slice3_mat)

        return CatLinearOperator(slice1, slice2, slice3, dim=-2)

    def evaluate_linear_op(self, linear_op):
        return self.psd_mat.detach().clone().requires_grad_()


class TestCatLinearOperatorColumn(LinearOperatorTestCase, unittest.TestCase):
    seed = 1

    def create_linear_op(self):
        root = torch.randn(6, 7)
        self.psd_mat = root.matmul(root.t())

        slice1_mat = self.psd_mat[:, :2].requires_grad_()
        slice2_mat = self.psd_mat[:, 2:4].requires_grad_()
        slice3_mat = self.psd_mat[:, 4:6].requires_grad_()

        slice1 = DenseLinearOperator(slice1_mat)
        slice2 = DenseLinearOperator(slice2_mat)
        slice3 = DenseLinearOperator(slice3_mat)

        return CatLinearOperator(slice1, slice2, slice3, dim=-1)

    def evaluate_linear_op(self, linear_op):
        return self.psd_mat.detach().clone().requires_grad_()


class TestCatLinearOperatorBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_op(self):
        root = torch.randn(3, 6, 7)
        self.psd_mat = root.matmul(root.mT)

        slice1_mat = self.psd_mat[..., :2, :].requires_grad_()
        slice2_mat = self.psd_mat[..., 2:4, :].requires_grad_()
        slice3_mat = self.psd_mat[..., 4:6, :].requires_grad_()

        slice1 = DenseLinearOperator(slice1_mat)
        slice2 = DenseLinearOperator(slice2_mat)
        slice3 = DenseLinearOperator(slice3_mat)

        return CatLinearOperator(slice1, slice2, slice3, dim=-2)

    def evaluate_linear_op(self, linear_op):
        return self.psd_mat.detach().clone().requires_grad_()


class TestCatLinearOperatorMultiBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    skip_slq_tests = True

    def create_linear_op(self):
        root = torch.randn(4, 3, 6, 7)
        self.psd_mat = root.matmul(root.mT)

        slice1_mat = self.psd_mat[..., :2, :].requires_grad_()
        slice2_mat = self.psd_mat[..., 2:4, :].requires_grad_()
        slice3_mat = self.psd_mat[..., 4:6, :].requires_grad_()

        slice1 = DenseLinearOperator(slice1_mat)
        slice2 = DenseLinearOperator(slice2_mat)
        slice3 = DenseLinearOperator(slice3_mat)

        return CatLinearOperator(slice1, slice2, slice3, dim=-2)

    def evaluate_linear_op(self, linear_op):
        return self.psd_mat.detach().clone().requires_grad_()


class TestCatLinearOperatorBatchCat(LinearOperatorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    skip_slq_tests = True

    def create_linear_op(self):
        root = torch.randn(5, 3, 6, 7)
        self.psd_mat = root.matmul(root.mT)

        slice1_mat = self.psd_mat[:2, ...].requires_grad_()
        slice2_mat = self.psd_mat[2:3, ...].requires_grad_()
        slice3_mat = self.psd_mat[3:, ...].requires_grad_()

        slice1 = DenseLinearOperator(slice1_mat)
        slice2 = DenseLinearOperator(slice2_mat)
        slice3 = DenseLinearOperator(slice3_mat)

        return CatLinearOperator(slice1, slice2, slice3, dim=0)

    def evaluate_linear_op(self, linear_op):
        return self.psd_mat.detach().clone().requires_grad_()

    def test_getitem_broadcasted_tensor_index(self):
        linear_op = self.create_linear_op()

        with self.assertRaises(RuntimeError):
            linear_op[torch.tensor([0, 1, 1]).view(-1, 1), ...]


class TestCatLinearOperatorSliceBounds(unittest.TestCase):
    def test_dense_slice_bounds_and_gradients(self) -> None:
        slices = [
            slice(None, 5),
            slice(1, 5),
            slice(None, 11),
            slice(-11, 4),
            slice(-11, 11),
            slice(4, 2),
            slice(None, 0),
            slice(6, None),
            slice(None, -5),
            slice(2, 2),
            slice(2, None),
            slice(1, -1),
        ]
        for dim in range(3):
            for selected in slices:
                with self.subTest(dim=dim, selected=selected):
                    first_shape, second_shape = [2, 3, 4], [2, 3, 4]
                    first_shape[dim], second_shape[dim] = 2, 3
                    first = torch.arange(torch.Size(first_shape).numel(), dtype=torch.float64).reshape(first_shape)
                    second = (100 + torch.arange(torch.Size(second_shape).numel(), dtype=torch.float64)).reshape(
                        second_shape
                    )
                    first.requires_grad_(True)
                    second.requires_grad_(True)
                    linear_op = CatLinearOperator(DenseLinearOperator(first), DenseLinearOperator(second), dim=dim)
                    indices = [slice(None)] * 3
                    indices[dim] = selected
                    actual = linear_op[tuple(indices)].to_dense()
                    expected = torch.cat((first, second), dim=dim)[tuple(indices)]
                    torch.testing.assert_close(actual, expected)

                    inputs = (first, second)
                    actual_grads = torch.autograd.grad(actual.sum(), inputs, allow_unused=True)
                    expected_grads = torch.autograd.grad(expected.sum(), inputs)
                    for source, actual_grad, expected_grad in zip(inputs, actual_grads, expected_grads):
                        if actual_grad is None:
                            actual_grad = torch.zeros_like(source)
                        torch.testing.assert_close(actual_grad, expected_grad)

    def test_nonempty_structured_slice_bounds(self) -> None:
        slices = [slice(None, 8), slice(1, 8), slice(None, 20), slice(-20, 7), slice(1, -1)]
        for operator in (DiagLinearOperator, IdentityLinearOperator):
            for dim in (0, 1):
                parts = (
                    [DiagLinearOperator(torch.arange(1.0, 5.0)), DiagLinearOperator(torch.arange(5.0, 9.0))]
                    if operator is DiagLinearOperator
                    else [IdentityLinearOperator(4), IdentityLinearOperator(4)]
                )
                linear_op = CatLinearOperator(*parts, dim=dim)
                dense = torch.cat([part.to_dense() for part in parts], dim=dim)
                for selected in slices:
                    with self.subTest(operator=operator.__name__, dim=dim, selected=selected):
                        indices = [slice(None), slice(None)]
                        indices[dim] = selected
                        torch.testing.assert_close(linear_op[tuple(indices)].to_dense(), dense[tuple(indices)])


if __name__ == "__main__":
    unittest.main()
