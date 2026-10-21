#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import (
    DiagLinearOperator,
    KroneckerProductLinearOperator,
    LinearOperator,
    RootLinearOperator,
)


class _NonlinearOperator(LinearOperator):
    def __init__(self, root, diagonal):
        super().__init__(root, diagonal)
        self.root, self.diagonal = root, diagonal

    def _matmul(self, rhs):
        return (self.root @ self.root.mT + self.diagonal.exp().diag_embed()) @ rhs

    def _size(self):
        return self.root.shape[:-1] + self.root.shape[-2:-1]

    def _transpose_nonbatch(self):
        return self


class TestBilinearDerivativeAutograd(unittest.TestCase):
    def _devices(self):
        return ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)

    def test_fallback_partial_derivative_and_mixed_higher_gradients(self):
        for device in self._devices():
            with self.subTest(device=device):
                root = torch.tensor([[0.3], [-0.2]], device=device, dtype=torch.double, requires_grad=True)
                diagonal = root[:, 0].square() - 0.7
                left = root.sin().expand(2, 3)
                right = root.cos().expand(2, 3)
                operator = _NonlinearOperator(root, diagonal)
                actual = operator._bilinear_derivative(left, right)
                product = left @ right.mT
                expected = ((product + product.mT) @ root, product.diagonal() * diagonal.exp())
                for observed, reference in zip(actual, expected):
                    torch.testing.assert_close(observed, reference)
                    self.assertTrue(observed.requires_grad)
                actual_higher = torch.autograd.grad(sum(g.square().sum() for g in actual), root, retain_graph=True)[0]
                expected_higher = torch.autograd.grad(sum(g.square().sum() for g in expected), root)[0]
                torch.testing.assert_close(actual_higher, expected_higher)

    def test_kronecker_composition_matches_dense_through_third_derivatives(self):
        for device in self._devices():
            for batch in ((), (2,), (2, 1)):
                with self.subTest(device=device, batch=batch):
                    root = torch.tensor([[0.3], [-0.2]], device=device, dtype=torch.double)
                    root = root.expand(*batch, 2, 1).clone().requires_grad_()
                    diagonal = root.new_tensor([-0.7, -0.4]).requires_grad_()
                    data_diagonal = root.new_tensor([0.8, 1.1, 0.9]).requires_grad_()
                    task_covariance = RootLinearOperator(root) + DiagLinearOperator(diagonal.exp())
                    operator = KroneckerProductLinearOperator(DiagLinearOperator(data_diagonal), task_covariance)
                    task = root @ root.mT + diagonal.exp().diag_embed()
                    dense = (data_diagonal.diag().reshape(3, 1, 3, 1) * task.unsqueeze(-3).unsqueeze(-2)).reshape(
                        *batch, 6, 6
                    )
                    torch.testing.assert_close(operator.to_dense(), dense)
                    parameters = (root, diagonal, data_diagonal)
                    rows = []
                    for result in (operator.to_dense(), dense):
                        first = torch.autograd.grad(
                            result.sin().sum(), parameters, create_graph=True, retain_graph=True
                        )
                        self.assertTrue(all(g.requires_grad for g in first))
                        second = torch.autograd.grad(
                            sum(g.square().sum() for g in first), parameters, create_graph=True, retain_graph=True
                        )
                        third = torch.autograd.grad(
                            sum(g.square().sum() for g in second), parameters, retain_graph=True
                        )
                        rows.append((first, second, third))
                    for observed, reference in zip(*rows):
                        for a, b in zip(observed, reference):
                            torch.testing.assert_close(a, b)

    def test_fallback_gradcheck_gradgradcheck_and_ordinary_backward(self):
        for device in self._devices():
            with self.subTest(device=device):
                root = torch.tensor([[0.3], [-0.2]], device=device, dtype=torch.double, requires_grad=True)
                diagonal = root.new_tensor([-0.7, -0.4], requires_grad=True)
                rhs = root.new_tensor([[0.4, 0.7], [-0.2, 0.3]], requires_grad=True)

                def apply(root, diagonal, rhs):
                    return _NonlinearOperator(root, diagonal).matmul(rhs)

                self.assertTrue(torch.autograd.gradcheck(apply, (root, diagonal, rhs)))
                self.assertTrue(torch.autograd.gradgradcheck(apply, (root, diagonal, rhs)))
                loss = apply(root, diagonal, rhs).square().sum()
                ordinary = torch.autograd.grad(loss, (root, diagonal, rhs))
                self.assertFalse(any(g.requires_grad for g in ordinary))


if __name__ == "__main__":
    unittest.main()
