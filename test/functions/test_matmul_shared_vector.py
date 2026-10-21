import unittest

import torch

from linear_operator import settings
from linear_operator.operators import CatLinearOperator, DenseLinearOperator


class TestBatchedVectorMatmulBackward(unittest.TestCase):
    def test_shared_vector_gradients_and_higher_derivatives(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        shapes = [(3, 4), (1, 4), (3, 1), (2, 3, 4), (2, 1, 4), (2, 3, 1), (2, 3, 2, 4)]
        for device in devices:
            for shape in shapes:
                for memory_efficient in (False, True):
                    with self.subTest(device=device, shape=shape, memory_efficient=memory_efficient):
                        torch.manual_seed(21)
                        matrix = torch.randn(shape, device=device, dtype=torch.double, requires_grad=True)
                        vector = torch.randn(shape[-1], device=device, dtype=torch.double, requires_grad=True)
                        params = (matrix, vector)
                        with settings.memory_efficient(memory_efficient):
                            actual = DenseLinearOperator(matrix) @ vector
                            expected = matrix @ vector
                            torch.testing.assert_close(actual, expected)
                            actual_loss = actual.sin().sum() + 0.3 * actual.square().sum()
                            expected_loss = expected.sin().sum() + 0.3 * expected.square().sum()
                            for order in range(3):
                                observed = torch.autograd.grad(actual_loss, params, create_graph=order < 2)
                                reference = torch.autograd.grad(expected_loss, params, create_graph=order < 2)
                                for a, b in zip(observed, reference):
                                    torch.testing.assert_close(a, b, rtol=2e-7, atol=2e-8)
                                if order < 2:
                                    actual_loss = sum(g.square().sum() for g in observed)
                                    expected_loss = sum(g.square().sum() for g in reference)

    def test_each_trainability_configuration_and_noncontiguous_vector(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for train_matrix, train_vector in ((True, False), (False, True), (True, True)):
                with self.subTest(device=device, train_matrix=train_matrix, train_vector=train_vector):
                    torch.manual_seed(23)
                    matrix = torch.randn(2, 3, 4, device=device, dtype=torch.double, requires_grad=train_matrix)
                    vector = torch.randn(8, device=device, dtype=torch.double)[::2].requires_grad_(train_vector)
                    weights = torch.randn(2, 3, device=device, dtype=torch.double)
                    params = tuple(p for p in (matrix, vector) if p.requires_grad)
                    actual = (DenseLinearOperator(matrix) @ vector * weights).sum()
                    expected = (matrix @ vector * weights).sum()
                    for a, b in zip(torch.autograd.grad(actual, params), torch.autograd.grad(expected, params)):
                        torch.testing.assert_close(a, b)

    def test_structured_batched_operator_vector_backward(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for dim in (-1, -2, -3):
                with self.subTest(device=device, dim=dim):
                    torch.manual_seed(25)
                    first = torch.randn(2, 3, 4, device=device, dtype=torch.double)
                    second = torch.randn_like(first)
                    dense = torch.cat((first, second), dim=dim)
                    vector = torch.randn(dense.shape[-1], device=device, dtype=torch.double, requires_grad=True)
                    operator = CatLinearOperator(DenseLinearOperator(first), DenseLinearOperator(second), dim=dim)
                    actual = operator @ vector
                    expected = dense @ vector
                    torch.testing.assert_close(actual, expected)
                    observed = torch.autograd.grad(actual.square().sum(), vector, create_graph=True)[0]
                    reference = torch.autograd.grad(expected.square().sum(), vector, create_graph=True)[0]
                    torch.testing.assert_close(observed, reference)
                    torch.testing.assert_close(
                        torch.autograd.grad(observed.square().sum(), vector)[0],
                        torch.autograd.grad(reference.square().sum(), vector)[0],
                    )

    def test_gradcheck_and_gradgradcheck(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for shape in ((2, 3, 4), (2, 1, 4), (2, 3, 1)):
                with self.subTest(device=device, shape=shape):
                    torch.manual_seed(27)
                    matrix = torch.randn(shape, device=device, dtype=torch.double, requires_grad=True)
                    vector = torch.randn(shape[-1], device=device, dtype=torch.double, requires_grad=True)

                    def evaluate(m, v):
                        return DenseLinearOperator(m) @ v

                    self.assertTrue(torch.autograd.gradcheck(evaluate, (matrix, vector), fast_mode=True))
                    self.assertTrue(torch.autograd.gradgradcheck(evaluate, (matrix, vector), fast_mode=True))
