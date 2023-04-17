from typing import Callable, Optional

import torch
from torch import Tensor

from .. import settings, utils
from ..operators import (
    IdentityLinearOperator,
    LinearOperator,
    LowRankRootLinearOperator,
    ZeroLinearOperator,
    to_linear_operator,
)
from .linear_solver import LinearSolver, LinearSolverState, LinearSystem


class CG(LinearSolver):
    r"""Conjugate gradient method.

    Iteratively solve linear systems of the form

    .. math:: Ax_* = b

    where :math:`A` is a symmetric positive-definite linear operator.

    :param abstol: Absolute residual tolerance.
    :param reltol: Relative residual tolerance.
    :max_iter: Maximum number of iterations. Defaults to `10 * rhs.shape[0]`.
    """

    def __init__(
        self,
        abstol: float = 1e-5,
        reltol: float = 1e-5,
        max_iter: int = None,
    ):
        self.abstol = abstol
        self.reltol = reltol
        self.max_iter = max_iter

    def solve(
        self,
        linear_op: LinearOperator,
        rhs: Tensor,
        /,
        *,
        x: Optional[Tensor] = None,
        precond: Optional[Callable] = None,
    ) -> LinearSolverState:
        r"""Solve linear system :math:`Ax_*=b`.

        :param linear_op: Linear operator :math:`A`.
        :param rhs: Right-hand-side :math:`b`.
        :param x: Initial guess :math:`x \approx x_*`.
        :param precond: Preconditioner :math:`P\approx A^{-1}`.
        """
        # Initialize solver state
        linear_op = to_linear_operator(linear_op)
        if self.max_iter is None:
            max_iter = 10 * rhs.shape[0]
        else:
            max_iter = self.max_iter

        if x is None:
            x = torch.zeros_like(rhs, requires_grad=True)
            inverse_op = ZeroLinearOperator(*linear_op.shape, dtype=linear_op.dtype, device=linear_op.device)
            residual = rhs
        else:
            residual = rhs - linear_op @ x

        if precond is None:
            precond = IdentityLinearOperator(diag_shape=linear_op.shape[1], dtype=linear_op.dtype)

        solver_state = LinearSolverState(
            problem=LinearSystem(A=linear_op, b=rhs),
            solution=x,
            forward_op=None,
            inverse_op=inverse_op,
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
            action = precond @ solver_state.residual
            linear_op_action = linear_op @ action

            # Observation
            observ = action.T @ solver_state.residual

            # Search direction
            if solver_state.iteration == 0:
                search_dir = action
            else:
                search_dir = (
                    action - solver_state.inverse_op @ linear_op_action
                )  # TODO: can be optimized for CG actions, at the cost of reorthogonalization

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
            solver_state.residual = (
                rhs - linear_op @ solver_state.solution
            )  # TODO: can be optimized for CG actions at the cost of potentially worsening residual approximation
            solver_state.residual_norm = torch.linalg.vector_norm(solver_state.residual, ord=2)

            # Update log-determinant
            solver_state.logdet = solver_state.logdet + torch.log(search_dir_sqnorm)

            # Update iteration
            solver_state.iteration += 1

        return solver_state


class CGGpytorch(LinearSolver):
    """Conjugate gradient method.

    Legacy implementation of the linear conjugate gradient method as originally used in GPyTorch.
    """

    def __init__(self, tol: float = 1e-4, max_iter: int = None):
        self.tol = tol
        self.max_iter = max_iter if max_iter is not None else settings.max_cg_iterations.value()

    def solve(
        self,
        linear_op: LinearOperator,
        rhs: Tensor,
        preconditioner: Optional[Callable] = None,
        num_tridiag: int = 0,
    ) -> LinearSolverState:
        return utils.linear_cg(
            linear_op._matmul,
            rhs,
            tolerance=self.tol,
            n_tridiag=num_tridiag,
            max_iter=self.max_iter,
            max_tridiag_iter=min(
                settings.max_lanczos_quadrature_iterations.value(), self.max_iter
            ),  # TODO: make lanczos iterations and preconditioner settable
            preconditioner=preconditioner,
        )
