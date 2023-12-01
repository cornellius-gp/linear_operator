from __future__ import annotations

from typing import Optional

import torch

from ...operators import LinearOperator
from .linear_solver_policy import LinearSolverPolicy


class GradientPolicy(LinearSolverPolicy):
    """Policy choosing (preconditioned) gradients / residuals :math:`Pr_i = P(b - Ax_i)` as actions.

    :param precond: Preconditioner :math:`P \\approx A^{-1}`.
    """

    def __init__(
        self,
        precond: Optional["LinearOperator"] = None,
        num_non_zero: Optional[int] = None,
    ) -> None:
        self.precond = precond
        self.num_nonzero = num_non_zero

        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        with torch.no_grad():
            action = solver_state.residual

            if isinstance(self.precond, (torch.Tensor, LinearOperator)):
                action = self.precond @ action
            elif callable(self.precond):
                action = self.precond(action).squeeze()

            if self.num_nonzero is not None:
                _, topk_idcs = torch.topk(torch.abs(action), k=self.num_nonzero, largest=True)
                sparse_action = torch.zeros(
                    solver_state.problem.A.shape[0],
                    dtype=solver_state.problem.A.dtype,
                    device=solver_state.problem.A.device,
                )
                sparse_action[topk_idcs] = action[topk_idcs]
                action = sparse_action

            return action
