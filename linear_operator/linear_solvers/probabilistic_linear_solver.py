from __future__ import annotations

from typing import Generator, Optional

import torch
from torch import Tensor

from .. import settings
from ..operators import (
    LinearOperator,
    LowRankRootLinearOperator,
    MulLinearOperator,
    ZeroLinearOperator,
    to_linear_operator,
)
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

    def solve_iterator(
        self,
        linear_op: LinearOperator,
        rhs: Tensor,
        /,
        x: Optional[Tensor] = None,
    ) -> Generator[LinearSolverState, None, None]:
        r"""Generator implementing the linear solver iteration.

        This function allows stepping through the solver iteration one step at a time and thus exposes internal quantities in the solver state cache.

        :param linear_op: Linear operator :math:`A`.
        :param rhs: Right-hand-side :math:`b`.
        :param x: Initial guess :math:`x \approx x_*`.
        """
        # Setup
        linear_op = to_linear_operator(linear_op)
        if self.max_iter is None:
            max_iter = 10 * rhs.shape[0]
        else:
            max_iter = self.max_iter

        # Ensure initial guess and rhs are vectors
        rhs = rhs.reshape(-1)
        if x is not None:
            x = x.reshape(-1)

        if x is None:
            x = torch.zeros_like(rhs, requires_grad=True)
            inverse_op = ZeroLinearOperator(
                *linear_op.shape, dtype=linear_op.dtype, device=linear_op.device
            )
            residual = rhs
            logdet = torch.zeros((), requires_grad=True)
        else:
            # Construct a better initial guess with a consistent inverse approximation such that x = inverse_op @ rhs
            action = x
            linear_op_action = linear_op @ action
            action_linear_op_action = torch.inner(linear_op_action, action)

            # Potentially improved initial guess x derived from initial guess
            step_size = torch.inner(action, rhs) / action_linear_op_action
            x = step_size * action

            # Initial residual
            linear_op_x = step_size * linear_op_action
            residual = rhs - linear_op_x

            # Consistent inverse approximation for new initial guess
            inverse_op = LowRankRootLinearOperator(
                (action / torch.sqrt(action_linear_op_action)).reshape(-1, 1)
            )

            # Log determinant
            logdet = torch.log(action_linear_op_action)

        # Initialize solver state
        solver_state = LinearSolverState(
            problem=LinearSystem(A=linear_op, b=rhs),
            solution=x,
            forward_op=None,
            inverse_op=inverse_op,
            residual=residual,
            residual_norm=torch.linalg.vector_norm(residual, ord=2),
            logdet=logdet,
            iteration=0,
            cache={
                "search_dir_sq_Anorms": [],
                "rhs_norm": torch.linalg.vector_norm(rhs, ord=2),
                "action": None,
                "observation": None,
                "search_dir": None,
                "step_size": None,
            },
        )

        yield solver_state

        while True:
            # Check convergence
            if (
                solver_state.residual_norm
                < max(self.abstol, self.reltol * solver_state.cache["rhs_norm"])
                or solver_state.iteration >= max_iter
            ):
                break

            # Select action
            action = self.policy(solver_state)
            linear_op_action = linear_op @ action

            # Observation
            observ = torch.inner(action, solver_state.residual)

            # Search direction
            if isinstance(solver_state.inverse_op, ZeroLinearOperator):
                search_dir = action
            else:
                search_dir = action - solver_state.inverse_op @ linear_op_action

            # Normalization constant
            search_dir_sqnorm = torch.inner(linear_op_action, search_dir)
            solver_state.cache["search_dir_sq_Anorms"].append(search_dir_sqnorm)

            if search_dir_sqnorm <= 0:
                if settings.verbose_linalg.on():
                    settings.verbose_linalg.logger.debug(
                        f"PLS terminated after {solver_state.iteration} iteration(s)"
                        + " due to a negative normalization constant."
                    )
                break

            # Update solution estimate
            step_size = observ / search_dir_sqnorm
            solver_state.solution = solver_state.solution + step_size * search_dir

            # Update inverse approximation
            if isinstance(solver_state.inverse_op, ZeroLinearOperator):
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
            solver_state.residual_norm = torch.linalg.vector_norm(
                solver_state.residual, ord=2
            )

            # Update log-determinant
            solver_state.logdet = solver_state.logdet + torch.log(search_dir_sqnorm)

            # Update iteration
            solver_state.iteration += 1

            # Update solver state cache
            solver_state.cache["action"] = action
            solver_state.cache["observation"] = observ
            solver_state.cache["search_dir"] = search_dir
            solver_state.cache["step_size"] = step_size

            yield solver_state

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

        solver_state = None

        for solver_state in self.solve_iterator(linear_op, rhs, x=x):
            pass

        return solver_state


class PLSsparse(LinearSolver):
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

    def solve_iterator(
        self,
        linear_op: LinearOperator,
        rhs: Tensor,
        /,
        x: Optional[Tensor] = None,
    ) -> Generator[LinearSolverState, None, None]:
        r"""Generator implementing the linear solver iteration.

        This function allows stepping through the solver iteration one step at a time and thus exposes internal quantities in the solver state cache.

        :param linear_op: Linear operator :math:`A`.
        :param rhs: Right-hand-side :math:`b`.
        :param x: Initial guess :math:`x \approx x_*`.
        """
        # Setup
        linear_op = to_linear_operator(linear_op)
        if self.max_iter is None:
            max_iter = 10 * rhs.shape[0]
        else:
            max_iter = self.max_iter

        # Ensure initial guess and rhs are vectors
        rhs = rhs.reshape(-1)
        if x is not None:
            x = x.reshape(-1)

        if x is None:
            x = torch.zeros_like(rhs, requires_grad=True)
            inverse_op = ZeroLinearOperator(
                *linear_op.shape, dtype=linear_op.dtype, device=linear_op.device
            )
            residual = rhs
            logdet = torch.zeros((), requires_grad=True)
        else:
            raise NotImplementedError(
                "Currently we do not support initializing with a given solution x."
            )

        # Initialize solver state
        solver_state = LinearSolverState(
            problem=LinearSystem(A=linear_op, b=rhs),
            solution=x,
            forward_op=None,
            inverse_op=inverse_op,
            residual=residual,
            residual_norm=torch.linalg.vector_norm(residual, ord=2),
            logdet=logdet,
            iteration=0,
            cache={
                "search_dir_sq_Anorms": [],
                "rhs_norm": torch.linalg.vector_norm(rhs, ord=2),
                "action": None,
                "actions": None,
                "linear_op_actions": None,
                "observation": None,
                "search_dir": None,
                "step_size": None,
                "compressed_solution": None,
            },
        )

        yield solver_state

        while True:
            # Check convergence
            if (
                solver_state.residual_norm
                < max(self.abstol, self.reltol * solver_state.cache["rhs_norm"])
                or solver_state.iteration >= max_iter
            ):
                break

            # Select action
            action = self.policy(solver_state)
            linear_op_action = linear_op @ action

            if solver_state.cache["actions"] is not None:
                prev_actions_linear_op_action = (
                    solver_state.cache["actions"].mT @ linear_op_action
                )
            else:
                prev_actions_linear_op_action = None

            # Observation
            observ = torch.inner(action, solver_state.residual)

            # Normalization constant
            action_linear_op_action = torch.inner(linear_op_action, action)

            search_dir_sqnorm = action_linear_op_action  # TODO: obsolete computation

            if solver_state.cache["actions"] is not None:
                gram_inv_tilde_z = torch.cholesky_solve(
                    prev_actions_linear_op_action.reshape(-1, 1),
                    solver_state.cache["cholfac_gram"],
                    upper=False,
                ).reshape(-1)

                search_dir_sqnorm = search_dir_sqnorm - torch.inner(
                    prev_actions_linear_op_action, gram_inv_tilde_z
                )  # TODO: obsolete computation

            solver_state.cache["search_dir_sq_Anorms"].append(search_dir_sqnorm)

            if search_dir_sqnorm <= 0:
                if settings.verbose_linalg.on():
                    settings.verbose_linalg.logger.debug(
                        f"PLS terminated after {solver_state.iteration} iteration(s)"
                        + " due to a negative normalization constant."
                    )
                break

            # Step size
            step_size = observ / search_dir_sqnorm  # TODO: obsolete

            if solver_state.cache["actions"] is None:
                # Matrix of previous actions
                solver_state.cache["actions"] = torch.reshape(action, (-1, 1))

                # Matrix of previous actions applied to the kernel matrix
                solver_state.cache["linear_op_actions"] = torch.reshape(
                    linear_op_action, (-1, 1)
                )

                # Initialize Cholesky factor
                solver_state.cache["cholfac_gram"] = action_linear_op_action.reshape(
                    1, 1
                )

            else:
                # Update to Cholesky factor of Gram matrix S_i'\hat{K}S_i
                new_cholfac_bottom_row_minus_last_entry = torch.linalg.solve_triangular(
                    solver_state.cache["cholfac_gram"],
                    prev_actions_linear_op_action.reshape(-1, 1),
                    upper=False,
                ).reshape(-1)
                new_cholfac_bottom_row_rightmost_entry = torch.sqrt(
                    action_linear_op_action
                    - torch.inner(
                        new_cholfac_bottom_row_minus_last_entry,
                        new_cholfac_bottom_row_minus_last_entry,
                    )
                )

                if new_cholfac_bottom_row_rightmost_entry.item() <= 0:
                    if settings.verbose_linalg.on():
                        settings.verbose_linalg.logger.debug(
                            f"PLS terminated after {solver_state.iteration} iteration(s)"
                            + " since the Cholesky factorization could not be updated."
                        )

                    break

                solver_state.cache["cholfac_gram"] = torch.vstack(
                    (
                        torch.hstack(
                            (
                                solver_state.cache["cholfac_gram"],
                                torch.zeros(
                                    (solver_state.iteration, 1),
                                    device=linear_op.device,
                                    dtype=linear_op.dtype,
                                ),
                            )
                        ),
                        torch.hstack(
                            (
                                new_cholfac_bottom_row_minus_last_entry,
                                new_cholfac_bottom_row_rightmost_entry,
                            )
                        ),
                    )
                )

                # TODO: Should we explicitly recompute the Cholesky factor here for stability? -> problem: O(i^2 max(i,k)) per iteration
                # Did not actually improve stability for subset Lanczos actions
                # solver_state.cache["cholfac_gram"] = torch.linalg.cholesky(
                #     solver_state.cache["linear_op_actions"].mT
                #     @ solver_state.cache["actions"],
                #     upper=False,
                # )

                # Matrix of actions
                solver_state.cache["actions"] = torch.hstack(
                    (solver_state.cache["actions"], action.reshape(-1, 1))
                )

                # Matrix of actions applied to the kernel matrix
                solver_state.cache["linear_op_actions"] = torch.hstack(
                    (
                        solver_state.cache["linear_op_actions"],
                        linear_op_action.reshape(-1, 1),
                    )
                )

            # Update compressed solution estimate
            solver_state.cache["compressed_solution"] = torch.cholesky_solve(
                (solver_state.cache["actions"].mT @ rhs).reshape(-1, 1),
                solver_state.cache["cholfac_gram"],
                upper=False,
            ).reshape(-1)

            # Update solution estimate
            solver_state.solution = (
                solver_state.cache["actions"]
                @ solver_state.cache["compressed_solution"]
            )

            # Update residual
            solver_state.cache["linear_op_actions_compressed_solution"] = (
                solver_state.cache["linear_op_actions"]
                @ solver_state.cache["compressed_solution"]
            )
            solver_state.residual = (
                solver_state.problem.b
                - solver_state.cache["linear_op_actions_compressed_solution"]
            )
            # TODO: Explicitly recomputing the residual improves stability a bit (for CG)
            #
            # solver_state.residual = (
            #     solver_state.problem.b - solver_state.problem.A @ solver_state.solution
            # )

            solver_state.residual_norm = torch.linalg.vector_norm(
                solver_state.residual, ord=2
            )
            # TODO: should we check for an increase in residual here to stop early?

            # Update inverse approximation
            solver_state.inverse_op = (
                None  # TODO: lazy representation for simpler code?
            )

            # Update log-determinant
            solver_state.logdet = solver_state.logdet + torch.log(search_dir_sqnorm)

            # Update iteration
            solver_state.iteration += 1

            # Update solver state cache
            solver_state.cache["action"] = action
            solver_state.cache["observation"] = observ
            solver_state.cache["step_size"] = step_size

            yield solver_state

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

        solver_state = None

        for solver_state in self.solve_iterator(linear_op, rhs, x=x):
            pass

        return solver_state
