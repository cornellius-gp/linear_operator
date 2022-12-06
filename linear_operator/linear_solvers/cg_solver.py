#!/usr/bin/env python3


import torch
from torch import Tensor

from .linear_solver import LinearSolver, SolverState
from ..operators import LinearOperator
from .. import utils
from .. import settings


class CGSolver(LinearSolver):
    """
    TODO
    """

    def __init__(self, tol: float = 1e-4, max_iter: int = None):
        # TODO: eventually atol and rtol
        self.tol = tol
        self.max_iter = max_iter if max_iter is not None else settings.max_cg_iterations.value()

    def solve(self, linear_op: LinearOperator, rhs: Tensor) -> SolverState:
        with torch.no_grad():
            preconditioner = linear_op.detach()._solve_preconditioner()
        return utils.linear_cg(
            linear_op._matmul,
            rhs,
            tolerance=self.tol,
            n_tridiag=0,
            max_iter=self.max_iter,
            max_tridiag_iter=settings.max_lanczos_quadrature_iterations.value(),
            preconditioner=preconditioner,
        )

