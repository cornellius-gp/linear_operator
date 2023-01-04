#!/usr/bin/env python3


from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor

from ..operators import LinearOperator

__all__ = ["LinearSolver", "LinearSolverState"]


@dataclass
class LinearSolverState:
    """State of a linear solver applied to :math:`Ax_*=b`.

    :param solution: Solution of the linear solve.
    :param forward_op: Estimate of the forward operation :math:`A`.
    :param inverse_op: Estimate of the inverse operation :math:`A^{-1}`.
    :param residual: Residual :math:`r_i = b - Ax_i`.
    :param residual_norm: Residual norm :math:`\\lVert r_i \\rVert_2`.
    :param logdet: Estimate of the log-determinant :math:`\\log \\operatorname{det}(A)`.
    :param iteration: Iteration of the solver.
    :param cache: Miscellaneous quantities cached by the solver.
    """

    solution: Tensor
    forward_op: LinearOperator
    inverse_op: LinearOperator
    residual: Tensor
    residual_norm: Tensor
    logdet: Tensor
    iteration: int
    cache: dict = None


class LinearSolver(ABC):
    """Abstract base class for linear solvers.

    Method which solves a linear system of the form

    .. math:: Ax_*=b.
    """

    @abstractmethod
    def solve(self, linear_op: LinearOperator, rhs: Tensor, /, **kwargs) -> LinearSolverState:
        raise NotImplementedError
