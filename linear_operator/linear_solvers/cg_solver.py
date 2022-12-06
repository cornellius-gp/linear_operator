#!/usr/bin/env python3


from typing import Callable, Optional

import torch
from torch import Tensor

from .. import settings, utils
from ..operators import LinearOperator
from .linear_solver import LinearSolver, SolverState


class CGSolver(LinearSolver):
    """
    TODO
    """

    def __init__(self, tol: float = 1e-4, max_iter: int = None):
        # TODO: eventually atol and rtol
        self.tol = tol
        self.max_iter = max_iter if max_iter is not None else settings.max_cg_iterations.value()

    def solve(
        self, linear_op: LinearOperator, rhs: Tensor, preconditioner: Optional[Callable] = None, num_tridiag: int = 0
    ) -> SolverState:
        return utils.linear_cg(
            linear_op._matmul,
            rhs,
            tolerance=self.tol,
            n_tridiag=num_tridiag,
            max_iter=self.max_iter,
            max_tridiag_iter=settings.max_lanczos_quadrature_iterations.value(),
            preconditioner=preconditioner,
        )
