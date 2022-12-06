#!/usr/bin/env python3


from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch import Tensor

from ..operators import LinearOperator


@dataclass
class SolverState:
    """
    TODO
    """

    solution: Tensor
    forward_op: LinearOperator
    inverse_op: LinearOperator
    residual: Tensor
    residual_norm: Tensor
    iteration: int


class LinearSolver(ABC):
    """
    TODO
    """

    @abstractmethod
    def solve(self, linear_op: LinearOperator, rhs: Tensor) -> SolverState:
        raise NotImplementedError


__all__ = ["LinearSolver", "SolverState"]
