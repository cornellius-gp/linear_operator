from typing import Callable, Optional

import torch
from torch import Tensor

from ..operators import LinearOperator, LowRankRootLinearOperator, to_linear_operator
from .linear_solver import LinearSolver, LinearSolverState


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
    :max_iter: Maximum number of iterations.
    """

    def __init__(
        self,
        policy: Callable,
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
        # Setup
        linear_op = to_linear_operator(linear_op)

        if x is None:
            x = torch.zeros_like(rhs)
            residual = rhs
        else:
            residual = rhs - linear_op @ x

        residual_norm = torch.linalg.vector_norm(residual, ord=2)
        inv_approx = None
        search_dir_sqnorm_list = []
        i = 0

        for i in range(self.max_iter):

            # Check convergence
            if (
                residual_norm < max(self.abstol, self.reltol * torch.linalg.vector_norm(rhs, ord=2))
                or i > self.max_iter
            ):
                break

            # Select action
            action = self.policy(
                x, linear_op, rhs, residual
            )  # TODO: should this operate on the state? -> allows caching, or use its own cache: action, policy_cache = self.policy(..., policy_cache)
            # TODO: policy cache / state is probably good design since it lets us keep it stateless
            linear_op_action = linear_op @ action

            # Observation
            observ = action.T @ residual

            # Search direction
            if i == 0:
                search_dir = action
            else:
                search_dir = action - inv_approx @ linear_op_action

            # Normalization constant
            search_dir_sqnorm = linear_op_action.T @ search_dir
            search_dir_sqnorm_list.append(search_dir_sqnorm)

            # Solution update
            x = x + observ / search_dir_sqnorm * search_dir

            # Inverse approximation
            if i == 0:
                inv_approx = LowRankRootLinearOperator((search_dir / torch.sqrt(search_dir_sqnorm)).reshape(-1, 1))
            else:
                inv_approx = LowRankRootLinearOperator(
                    torch.concat(
                        (
                            inv_approx.root.to_dense(),
                            (search_dir / torch.sqrt(search_dir_sqnorm)).reshape(-1, 1),
                        ),
                        dim=1,
                    )
                )

            # Compute residual
            residual = rhs - linear_op @ x
            residual_norm = torch.linalg.vector_norm(residual, ord=2)

        # Output
        search_dir_sq_Anorms = torch.as_tensor(search_dir_sqnorm_list)

        return LinearSolverState(
            solution=x,
            forward_op=linear_op @ inv_approx @ linear_op if inv_approx is not None else None,
            inverse_op=inv_approx,
            residual=residual,
            residual_norm=residual_norm,
            logdet=torch.sum(torch.log(search_dir_sq_Anorms)),
            iteration=i,
            cache={
                "search_dir_sq_Anorms": search_dir_sq_Anorms,
            },
        )
