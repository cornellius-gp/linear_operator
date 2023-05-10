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

        # action = torch.randn(
        #     solver_state.problem.A.shape[1],
        #     dtype=solver_state.problem.A.dtype,
        #     device=solver_state.problem.A.device,
        # )
        # action = action.div(torch.linalg.vector_norm(action))

        # if self.num_nonzero is not None:
        #     topk_vals, topk_idcs = torch.topk(
        #         torch.abs(action), k=self.num_nonzero, largest=True
        #     )
        #     action[topk_idcs] = topk_vals

        #     assert sum(action > 0.0) == self.num_nonzero

        # return action


# TODO: try adverserial policy for better gradients?
