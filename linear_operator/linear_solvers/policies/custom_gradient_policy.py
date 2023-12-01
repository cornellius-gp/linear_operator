from __future__ import annotations

from typing import Optional

import torch

from ...operators import LinearOperator
from .linear_solver_policy import LinearSolverPolicy


class CustomGradientPolicy(LinearSolverPolicy):
    def __init__(
        self,
        linop: LinearOperator,
        rhs: torch.Tensor,
        num_non_zero: Optional[int] = None,
    ) -> None:
        self.linop = linop
        self.rhs = rhs
        self.num_nonzero = num_non_zero

        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        with torch.no_grad():
            action = self.rhs - self.linop @ solver_state.solution

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
