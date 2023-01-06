from __future__ import annotations

from typing import Optional

import torch

from ...utils.lanczos import lanczos_tridiag, lanczos_tridiag_to_diag
from .linear_solver_policy import LinearSolverPolicy


class LanczosPolicy(LinearSolverPolicy):
    """Policy choosing approximate eigenvectors as actions."""

    def __init__(self, descending: bool = True, max_iter: Optional[int] = None) -> None:
        self.descending = descending
        self.max_iter = max_iter
        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        if solver_state.iteration == 0:
            # Compute approximate eigenvectors
            Q, T = lanczos_tridiag(
                solver_state.problem.A.matmul,
                max_iter=solver_state.problem.A.shape[1] if self.max_iter is None else self.max_iter,
                dtype=solver_state.problem.A.dtype,
                device=solver_state.problem.A.device,
                matrix_shape=solver_state.problem.A.shape,
                tol=1e-5,
            )
            evals_lanczos, evecs_T = lanczos_tridiag_to_diag(T)
            evecs_lanczos = Q @ evecs_T

            # Cache approximate eigenvectors
            solver_state.cache["evals_lanczos"], idcs = torch.sort(evals_lanczos, descending=self.descending)
            solver_state.cache["evecs_lanczos"] = evecs_lanczos[:, idcs]

        # Return approximate eigenvectors according to strategy
        if solver_state.iteration < solver_state.cache["evecs_lanczos"].shape[1]:
            return solver_state.cache["evecs_lanczos"][:, solver_state.iteration]
        else:
            return solver_state.residual
