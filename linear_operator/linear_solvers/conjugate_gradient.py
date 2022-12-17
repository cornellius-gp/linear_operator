#!/usr/bin/env python3


from typing import Callable, Optional

import torch
from torch import Tensor

from .. import settings, utils
from ..operators import (
    IdentityLinearOperator,
    LinearOperator,
    LowRankRootLinearOperator,
)
from .linear_solver import LinearSolver, LinearSolverState


class CG(LinearSolver):
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
    ) -> LinearSolverState:
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

        inv_approx = None
        search_dir_sqnorm_list = []

        for i in range(self.max_iter):

            # Compute residual
            residual = (
                rhs - linear_op @ x
            )  # TODO: can be optimized for CG actions at the cost of potentially worsening residual approximation

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
            if i == 0:
                search_dir = action
            else:
                search_dir = (
                    action - inv_approx @ linear_op_action
                )  # TODO: can be optimized for CG actions, at the cost of reorthogonalization

            # Normalization constant
            search_dir_sqnorm = linear_op_action.T @ search_dir
            search_dir_sqnorm_list.append(search_dir_sqnorm)

            # Solution update
            x = x + observ / search_dir_sqnorm * search_dir

            # Inverse approximation
            if i == 0:
                inv_approx = LowRankRootLinearOperator((search_dir / torch.sqrt(search_dir_sqnorm)).reshape(-1, 1))
            else:
                inv_approx = LowRankRootLinearOperator(
                    torch.concat(
                        (
                            inv_approx.root.to_dense(),
                            (search_dir / torch.sqrt(search_dir_sqnorm)).reshape(-1, 1),
                        ),
                        dim=1,
                    )
                )

        return x, inv_approx, torch.as_tensor(search_dir_sqnorm_list)


class CGGpytorch(LinearSolver):
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
    ) -> LinearSolverState:
        return utils.linear_cg(
            linear_op._matmul,
            rhs,
            tolerance=self.tol,
            n_tridiag=num_tridiag,
            max_iter=self.max_iter,
            max_tridiag_iter=min(settings.max_lanczos_quadrature_iterations.value(), self.max_iter),
            preconditioner=preconditioner,
        )
