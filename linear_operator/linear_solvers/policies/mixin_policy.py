from __future__ import annotations

import torch

from .linear_solver_policy import LinearSolverPolicy


class MixinPolicy(LinearSolverPolicy):
    """Policy mixing actions from two policies based on a coefficient."""

    def __init__(self, base_policy: LinearSolverPolicy, mixin_policy: LinearSolverPolicy, mixin_coeff: float) -> None:
        self.base_policy = base_policy
        self.mixin_policy = mixin_policy
        self.mixin_coeff = mixin_coeff

        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        base_action = self.base_policy(solver_state)
        mixin_action = self.mixin_policy(solver_state)

        return (1.0 - self.mixin_coeff) * base_action + self.mixin_coeff * mixin_action
