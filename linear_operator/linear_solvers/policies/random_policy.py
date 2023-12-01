from __future__ import annotations

from typing import Optional

import torch

from .linear_solver_policy import LinearSolverPolicy


class RandomPolicy(LinearSolverPolicy):
    def __init__(
        self,
        num_non_zero: Optional[int] = None,
    ) -> None:
        self.num_nonzero = num_non_zero

        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        action = torch.zeros(
            solver_state.problem.A.shape[0],
            dtype=solver_state.problem.A.dtype,
            device=solver_state.problem.A.device,
        )

        with torch.no_grad():
            perm = torch.randperm(solver_state.problem.A.shape[0])
            idcs = perm[: self.num_nonzero]

        action[idcs] = 1.0

        return action
