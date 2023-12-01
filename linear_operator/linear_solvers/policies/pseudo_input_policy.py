from __future__ import annotations

from typing import Optional

import gpytorch
import torch

from .linear_solver_policy import LinearSolverPolicy


class PseudoInputPolicy(LinearSolverPolicy):
    """Policy choosing kernel functions evaluated at (pseudo-) inputs as actions.

    .. math :: s_i = k(X, z_i)

    :param precond: Preconditioner :math:`P \\approx A^{-1}`.
    """

    def __init__(
        self,
        kernel: gpytorch.kernels.Kernel,
        train_data: torch.Tensor,
        pseudo_inputs: torch.Tensor,
        sparsification_threshold: float = 0.0,
        num_non_zero: Optional[int] = None,
    ) -> None:
        self.kernel = kernel
        self.kernel.to(device=train_data.device, dtype=train_data.dtype)
        self.train_data = train_data
        self.pseudo_inputs = pseudo_inputs
        self.sparsification_threshold = sparsification_threshold
        self.num_nonzero = num_non_zero
        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        with torch.no_grad():
            action = (
                self.kernel(
                    self.train_data,
                    self.pseudo_inputs[solver_state.iteration].reshape(1, -1),
                )
                .evaluate_kernel()
                .to_dense()
            ).reshape(-1)

            if self.sparsification_threshold > 0.0:
                action[torch.abs(action) < self.sparsification_threshold] = 0.0

            if self.num_nonzero is not None:
                _, topk_idcs = torch.topk(torch.abs(action), k=self.num_nonzero, largest=True)
                sparse_action = torch.zeros(
                    solver_state.problem.A.shape[0],
                    dtype=solver_state.problem.A.dtype,
                    device=solver_state.problem.A.device,
                )
                sparse_action[topk_idcs] = action[topk_idcs]
                action = sparse_action

        return action
