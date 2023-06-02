#!/usr/bin/env python3

import unittest

import torch

from linear_operator.operators import (
    KernelLinearOperator,
    KroneckerProductLinearOperator,
    MatmulLinearOperator,
    RootLinearOperator,
)
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase, RectangularLinearOperatorTestCase


def _covar_func(x1, x2, lengthscale, outputscale):
    # RBF kernel function
    # x1: ... x N x D
    # x2: ... x M x D
    # lengthscale: ... x 1 x D
    # outputscale: ...
    lengthscale = lengthscale.mean(dim=-3)  # Remove extraneous dimension added for testing
    x1 = x1.div(lengthscale)
    x2 = x2.div(lengthscale)
    sq_dist = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).square().sum(dim=-1)
    kern = sq_dist.div(-2.0).exp().mul(outputscale[..., None, None].square())
    return kern


def _nystrom_covar_func(x1, x2, lengthscale, outputscale, inducing_points):
    # RBF kernel function w/ Nystrom approximation
    # x1: ... x N x D
    # x2: ... x M x D
    # lengthscale: ... x 1 x D
    # outputscale: ...
    ones = torch.ones_like(outputscale)
    K_zz_chol = _covar_func(inducing_points, inducing_points, lengthscale, ones)
    K_zx1 = _covar_func(inducing_points, x1, lengthscale, ones)
    K_zx2 = _covar_func(inducing_points, x2, lengthscale, ones)
    kern = MatmulLinearOperator(
        outputscale[..., None, None] * torch.linalg.solve_triangular(K_zz_chol, K_zx1, upper=False).mT,
        outputscale[..., None, None] * torch.linalg.solve_triangular(K_zz_chol, K_zx2, upper=False),
    )
    return kern


def _multitask_covar_func(x1, x2, lengthscale, outputscale, lmc_coeffs):
    # RBF kernel function w/ Nystrom approximation
    # x1: ... x N x D
    # x2: ... x M x D
    # lengthscale: ... x 1 x D
    # outputscale: ...
    K_xx = _covar_func(x1, x2, lengthscale=lengthscale, outputscale=outputscale)
    return KroneckerProductLinearOperator(K_xx, RootLinearOperator(lmc_coeffs))


class TestKernelLinearOperatorRectangular(RectangularLinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_op(self):
        x1 = torch.randn(3, 1, 5, 6)
        x2 = torch.randn(2, 4, 6)
        lengthscale = torch.nn.Parameter(torch.ones(4, 1, 6))
        # Adding an extraneous -3 dimension to test functionality
        outputscale = torch.nn.Parameter(torch.ones(3, 2))
        return KernelLinearOperator(
            x1,
            x2,
            lengthscale=lengthscale,
            outputscale=outputscale,
            covar_func=_covar_func,
            num_nonbatch_dimensions={"lengthscale": 3, "outputscale": 0},
        )

    def evaluate_linear_op(self, linop):
        return _covar_func(linop.x1, linop.x2, **linop.tensor_params)


class TestKernelLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_op(self):
        x = torch.randn(3, 5, 6)
        lengthscale = torch.nn.Parameter(torch.ones(3, 4, 1, 6))
        # Adding an extraneous -3 dimension to test functionality
        outputscale = torch.nn.Parameter(torch.ones(2, 1))
        return KernelLinearOperator(
            x,
            x,
            lengthscale=lengthscale,
            outputscale=outputscale,
            covar_func=_covar_func,
            num_nonbatch_dimensions={"lengthscale": 3, "outputscale": 0},
        )

    def evaluate_linear_op(self, linop):
        return _covar_func(linop.x1, linop.x2, **linop.tensor_params)


class TestKernelLinearOperatorRectangularLinOpReturn(TestKernelLinearOperatorRectangular, unittest.TestCase):
    seed = 0

    def create_linear_op(self):
        x1 = torch.randn(3, 4, 6)
        x2 = torch.randn(3, 5, 6)
        inducing_points = torch.randn(3, 6)
        lengthscale = torch.nn.Parameter(torch.ones(3, 4, 1, 6))
        # Adding an extraneous -3 dimension to test functionality
        outputscale = torch.nn.Parameter(torch.ones(2, 1))
        return KernelLinearOperator(
            x1,
            x2,
            lengthscale=lengthscale,
            outputscale=outputscale,
            inducing_points=inducing_points,
            covar_func=_nystrom_covar_func,
            num_nonbatch_dimensions={"lengthscale": 3, "outputscale": 0},
        )

    def evaluate_linear_op(self, linop):
        return _nystrom_covar_func(linop.x1, linop.x2, **linop.tensor_params).to_dense()


class TestKernelLinearOperatorLinOpReturn(TestKernelLinearOperator, unittest.TestCase):
    seed = 0

    def create_linear_op(self):
        x = torch.randn(3, 4, 6)
        inducing_points = torch.randn(20, 6)  # Overparameterized nystrom approx for invertibility
        lengthscale = torch.nn.Parameter(torch.ones(3, 4, 1, 6))
        # Adding an extraneous -3 dimension to test functionality
        outputscale = torch.nn.Parameter(torch.ones(2, 1))
        return KernelLinearOperator(
            x,
            x,
            lengthscale=lengthscale,
            outputscale=outputscale,
            inducing_points=inducing_points,
            covar_func=_nystrom_covar_func,
            num_nonbatch_dimensions={"lengthscale": 3, "outputscale": 0},
        )

    def evaluate_linear_op(self, linop):
        return _nystrom_covar_func(linop.x1, linop.x2, **linop.tensor_params).to_dense()


class TestKernelLinearOperatorMultiOutput(TestKernelLinearOperator, unittest.TestCase):
    seed = 0

    def create_linear_op(self):
        x = torch.randn(3, 4, 6)
        lengthscale = torch.nn.Parameter(torch.ones(3, 4, 1, 6))
        # Adding an extraneous -3 dimension to test functionality
        outputscale = torch.nn.Parameter(torch.ones(2, 1))
        lmc_coeffs = torch.nn.Parameter(torch.tensor([[1.0, 0.5], [0.5, 1.0]]))
        return KernelLinearOperator(
            x,
            x,
            lengthscale=lengthscale,
            outputscale=outputscale,
            lmc_coeffs=lmc_coeffs,
            covar_func=_multitask_covar_func,
            num_outputs_per_input=(2, 2),
            num_nonbatch_dimensions={"lengthscale": 3, "outputscale": 0},
        )

    def evaluate_linear_op(self, linop):
        return _multitask_covar_func(linop.x1, linop.x2, **linop.tensor_params).to_dense()
