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
        frequency_order: str = "interleaved",
        phase: float = -0.5 * pi,
        num_non_zero: Optional[int] = None,
    ) -> None:
        self.frequency_order = frequency_order
        self.phase = phase
        self.num_nonzero = num_non_zero

        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        with torch.no_grad():
            frequencies = torch.linspace(
                0,
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

            ts = torch.linspace(
                0,
                1,
                solver_state.problem.A.shape[1],
                dtype=solver_state.problem.A.dtype,
                device=solver_state.problem.A.device,
            )
            # TODO: this completely ignores the order of the datapoints, and therefore doesnt reflect sine waves in input space
            action = torch.sin(
                2 * pi * frequencies[solver_state.iteration] * ts + self.phase
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
