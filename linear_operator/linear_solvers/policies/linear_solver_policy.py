from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class LinearSolverPolicyState:
    """State of the policy of a linear solver."""

    iteration: int


class LinearSolverPolicy(ABC):
    """Policy of a linear solver.

    A linear solver policy chooses actions to observe the linear system :math:`Ax_* = b`.
    """

    @abstractmethod
    def __call__(self, solver_state: LinearSolverState) -> Tuple[torch.Tensor, LinearSolverState]:
        """Generate an action.

        :param solver_state:
        """
        raise NotImplementedError
