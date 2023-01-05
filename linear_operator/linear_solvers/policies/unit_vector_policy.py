from __future__ import annotations

import torch

from .linear_solver_policy import LinearSolverPolicy


class UnitVectorPolicy(LinearSolverPolicy):
    """Policy choosing unit vectors as actions."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        action = torch.zeros_like(solver_state.solution)
        action[solver_state.iteration] = 1.0
        return action
