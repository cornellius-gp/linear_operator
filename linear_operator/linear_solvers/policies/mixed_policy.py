from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch

from .linear_solver_policy import LinearSolverPolicy


class MixedPolicy(LinearSolverPolicy):
    """Policy choosing mixed actions."""

    def __init__(
        self, policies: Iterable[LinearSolverPolicy], max_policy_iters: Iterable[int]
    ) -> None:
        self.policies = np.asarray(policies)
        self.max_policy_iters = np.asarray(max_policy_iters)

        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        current_policy = self.policies[self.max_policy_iters > solver_state.iteration][
            0
        ]
        return current_policy(solver_state)
