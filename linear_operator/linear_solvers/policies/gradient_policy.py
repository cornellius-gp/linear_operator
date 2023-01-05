from __future__ import annotations

from typing import Optional

import torch

from .linear_solver_policy import LinearSolverPolicy


class GradientPolicy(LinearSolverPolicy):
    """Policy choosing (preconditioned) gradients / residuals :math:`Pr_i = P(b - Ax_i)` as actions.

    :param precond: Preconditioner :math:`P \\approx A^{-1}`.
    """

    def __init__(self, precond: Optional["LinearOperator"] = None) -> None:
        self.precond = precond
        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        if self.precond is None:
            return solver_state.residual
        return self.precond @ solver_state.residual
