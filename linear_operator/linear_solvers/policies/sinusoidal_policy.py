from __future__ import annotations

from math import pi
from typing import Optional

import torch

from ...operators import LinearOperator
from .linear_solver_policy import LinearSolverPolicy


class SinusoidalPolicy(LinearSolverPolicy):
    """Policy returning sinusoidal actions with different frequencies."""

    def __init__(
        self,
        train_data: torch.Tensor,
        frequency_order: str = "interleaved",
        phase: float = -0.5 * pi,
        num_non_zero: Optional[int] = None,
    ) -> None:
        self.train_data = train_data
        self.frequency_order = frequency_order
        self.phase = phase
        self.num_nonzero = num_non_zero

        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        with torch.no_grad():
            frequencies = torch.linspace(
                1e-3,
                1,
                solver_state.problem.A.shape[1],
                dtype=solver_state.problem.A.dtype,
                device=solver_state.problem.A.device,
            )
            if self.frequency_order == "ascending":
                pass
            elif self.frequency_order == "descending":
                frequencies = torch.flipud(frequencies)
            elif self.frequency_order == "interleaved":
                frequencies = torch.hstack(
                    (
                        frequencies[:, None],
                        torch.flipud(frequencies)[:, None],
                    )
                ).flatten()

            action = torch.sin(
                2
                * pi
                * frequencies[solver_state.iteration]
                * self.train_data.sum(dim=1)
                + self.phase
            )

            if self.num_nonzero is not None:
                topk_vals, topk_idcs = torch.topk(
                    torch.abs(action), k=self.num_nonzero, largest=True, sorted=False
                )
                action = torch.zeros(
                    solver_state.problem.A.shape[0],
                    dtype=solver_state.problem.A.dtype,
                    device=solver_state.problem.A.device,
                )
                action[topk_idcs] = topk_vals

            return action
