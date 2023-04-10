from __future__ import annotations

from typing import Optional

import torch

from ...operators import LinearOperator
from .linear_solver_policy import LinearSolverPolicy


class MixedPolicy(LinearSolverPolicy):
    """Policy choosing mixed actions."""

    def __init__(self, precond: Optional["LinearOperator"] = None) -> None:
        self.precond = precond
        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:

        if solver_state.iteration == 0:

            init_vec = torch.randn(
                solver_state.problem.A.shape[1],
                dtype=solver_state.problem.A.dtype,
                device=solver_state.problem.A.device,
            )
            init_vec = init_vec.div(torch.linalg.vector_norm(init_vec))

            # Cache initial vector
            solver_state.cache["init_vec"] = init_vec

        if solver_state.iteration % 10 == 0 and solver_state.iteration != 0:
            return solver_state.residual
        else:
            return solver_state.cache["init_vec"] - solver_state.problem.A @ solver_state.solution
