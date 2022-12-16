#!/usr/bin/env python3


from typing import Callable, Optional

import torch
from torch import Tensor

from .. import settings, utils
from ..operators import IdentityLinearOperator, LinearOperator, LowRankRootLinearOperator, ZeroLinearOperator
from .linear_solver import LinearSolver, SolverState


class IterGPCGSolver(LinearSolver):
    r"""Conjugate gradient method.

    Iteratively solve linear systems of the form

    .. math:: Ax_* = b

    where :math:`A` is a symmetric positive-definite linear operator.

    :param abstol: Absolute residual tolerance.
    :param reltol: Relative residual tolerance.
    :max_iter: Maximum number of iterations.
    """

    def __init__(
        self,
        abstol: float = 1e-5,
        reltol: float = 1e-5,
        max_iter: int = None,
    ):
        self.abstol = abstol
        self.reltol = reltol
        self.max_iter = max_iter if max_iter is not None else settings.max_cg_iterations.value()

    def solve(
        self,
        linear_op: LinearOperator,
        rhs: Tensor,
        /,
        *,
        x: Optional[Tensor] = None,
        preconditioner: Optional[Callable] = None,
    ) -> SolverState:
        r"""Solve linear system :math:`Ax_*=b`.

        :param linear_op: Linear operator :math:`A`.
        :param rhs: Right-hand-side :math:`b`.
        :param x: Initial guess :math:`x \approx x_*`.
        :param preconditioner: Preconditioner.
        """

        # Setup
        if preconditioner is None:
            preconditioner = IdentityLinearOperator(diag_shape=linear_op.shape[1], dtype=linear_op.dtype)

        if x is None:
            x = torch.zeros_like(rhs)

        inv_approx = ZeroLinearOperator(*linear_op.shape, dtype=linear_op.dtype)

        for i in range(self.max_iter):

            # Compute residual
            residual = rhs - linear_op @ x

            # Check convergence
            residual_norm = torch.linalg.norm(residual, ord=2)
            if residual_norm < max(self.abstol, self.reltol * torch.linalg.norm(rhs, ord=2)) or i > self.max_iter:
                break

            # Select action
            action = preconditioner @ residual
            linear_op_action = linear_op @ action

            # Observation
            observ = action.T @ residual

            # Search direction
            search_dir = action - inv_approx @ linear_op_action

            # Normalization constant
            searchdir_norm = linear_op_action.T @ search_dir
            normalized_searchdir = search_dir / searchdir_norm

            # Solution update
            x = x + observ * normalized_searchdir

            # Inverse approximation
            inv_approx = LowRankRootLinearOperator(torch.concat((inv_approx.root, normalized_searchdir), dim=1))

        return x, inv_approx


class CGSolver(LinearSolver):
    """Conjugate gradient method.
    TODO
    """

    def __init__(self, tol: float = 1e-4, max_iter: int = None):
        self.tol = tol
        self.max_iter = max_iter if max_iter is not None else settings.max_cg_iterations.value()

    def solve(
        self,
        linear_op: LinearOperator,
        rhs: Tensor,
        preconditioner: Optional[Callable] = None,
        num_tridiag: int = 0,
    ) -> SolverState:
        return utils.linear_cg(
            linear_op._matmul,
            rhs,
            tolerance=self.tol,
            n_tridiag=num_tridiag,
            max_iter=self.max_iter,
            max_tridiag_iter=min(settings.max_lanczos_quadrature_iterations.value(), self.max_iter),
            preconditioner=preconditioner,
        )
