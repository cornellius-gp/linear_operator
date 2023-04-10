from __future__ import annotations

from typing import Optional

import torch

from ...operators import LinearOperator
from .linear_solver_policy import LinearSolverPolicy


class GradientPolicy(LinearSolverPolicy):
    """Policy choosing (preconditioned) gradients / residuals :math:`Pr_i = P(b - Ax_i)` as actions.

    :param precond: Preconditioner :math:`P \\approx A^{-1}`.
    """

    def __init__(self, precond: Optional["LinearOperator"] = None) -> None:
        self.precond = precond
        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:

        action = solver_state.residual

        if isinstance(self.precond, (torch.Tensor, LinearOperator)):
            action = self.precond @ action
        elif callable(self.precond):
            action = self.precond(action).squeeze()

        return action
