from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..operators import LinearOperator, LowRankRootLinearOperator, to_linear_operator
from .linear_solver import LinearSolver, LinearSolverState, LinearSystem


class PLS(LinearSolver):
    """Probabilistic linear solver.

    Iteratively solve linear systems of the form

    .. math:: Ax_* = b

    where :math:`A` is a (symmetric positive-definite) linear operator. A probabilistic
    linear solver chooses actions :math:`s_i` in each iteration to observe the residual
    by computing :math:`\\alpha_i = s_i^\\top (b - Ax_i)`.

    :param policy: Policy selecting actions :math:`s_i` to probe the residual with.
    :param abstol: Absolute residual tolerance.
    :param reltol: Relative residual tolerance.
    :max_iter: Maximum number of iterations. Defaults to `10 * rhs.shape[0]`.
    """

    def __init__(
        self,
        policy: "LinearSolverPolicy",
        abstol: float = 1e-5,
        reltol: float = 1e-5,
        max_iter: int = None,
    ):
        self.policy = policy
        self.abstol = abstol
        self.reltol = reltol
        self.max_iter = max_iter

    def solve(
        self,
        linear_op: LinearOperator,
        rhs: Tensor,
        /,
        x: Optional[Tensor] = None,
    ) -> LinearSolverState:
        r"""Solve linear system :math:`Ax_*=b`.

        :param linear_op: Linear operator :math:`A`.
        :param rhs: Right-hand-side :math:`b`.
        :param x: Initial guess :math:`x \approx x_*`.
        """
        # Initialize solver state
        linear_op = to_linear_operator(linear_op)
        if self.max_iter is None:
            max_iter = 10 * rhs.shape[0]
        else:
            max_iter = self.max_iter

        if x is None:
            x = torch.zeros_like(rhs, requires_grad=True)
            residual = rhs
        else:
            residual = rhs - linear_op @ x

        solver_state = LinearSolverState(
            problem=LinearSystem(A=linear_op, b=rhs),
            solution=x,
            forward_op=None,
            inverse_op=None,
            residual=residual,
            residual_norm=torch.linalg.vector_norm(residual, ord=2),
            logdet=torch.zeros((), requires_grad=True),
            iteration=0,
            cache={
                "search_dir_sq_Anorms": [],
                "rhs_norm": torch.linalg.vector_norm(rhs, ord=2),
            },
        )

        while (  # Check convergence
            solver_state.residual_norm > max(self.abstol, self.reltol * solver_state.cache["rhs_norm"])
            and solver_state.iteration < max_iter
        ):

            # Select action
            action = self.policy(solver_state)
            linear_op_action = linear_op @ action

            # Observation
            observ = action.T @ solver_state.residual

            # Search direction
            if solver_state.iteration == 0:
                search_dir = action
            else:
                search_dir = action - solver_state.inverse_op @ linear_op_action

            # Normalization constant
            search_dir_sqnorm = linear_op_action.T @ search_dir
            solver_state.cache["search_dir_sq_Anorms"].append(search_dir_sqnorm)

            # Update solution estimate
            solver_state.solution = solver_state.solution + observ / search_dir_sqnorm * search_dir

            # Update inverse approximation
            if solver_state.iteration == 0:
                solver_state.inverse_op = LowRankRootLinearOperator(
                    (search_dir / torch.sqrt(search_dir_sqnorm)).reshape(-1, 1)
                )
            else:
                solver_state.inverse_op = LowRankRootLinearOperator(
                    torch.concat(
                        (
                            solver_state.inverse_op.root.to_dense(),
                            (search_dir / torch.sqrt(search_dir_sqnorm)).reshape(-1, 1),
                        ),
                        dim=1,
                    )
                )

            # Update residual
            solver_state.residual = rhs - linear_op @ solver_state.solution
            solver_state.residual_norm = torch.linalg.vector_norm(solver_state.residual, ord=2)

            # Update log-determinant
            solver_state.logdet = solver_state.logdet + torch.log(search_dir_sqnorm)

            # Update iteration
            solver_state.iteration += 1

        # Finalize solver state
        solver_state.forward_op = (
            linear_op @ solver_state.inverse_op @ linear_op if solver_state.inverse_op is not None else None
        )

        return solver_state
