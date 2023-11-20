#!/usr/bin/env python3
from typing import Callable, Optional, Tuple, Union

import torch
from jaxtyping import Float
from torch import Tensor

from linear_operator.operators import to_dense
from linear_operator.operators._linear_operator import LinearOperator
from linear_operator.operators.added_diag_linear_operator import AddedDiagLinearOperator
from linear_operator.operators.diag_linear_operator import ConstantDiagLinearOperator, DiagLinearOperator
from linear_operator.operators.low_rank_root_linear_operator import LowRankRootLinearOperator
from linear_operator.operators.sum_batch_linear_operator import SumBatchLinearOperator

from linear_operator.utils.cholesky import psd_safe_cholesky
from linear_operator.utils.memoize import cached


class LowRankRootAddedDiagLinearOperator(AddedDiagLinearOperator):
    def __init__(self, *linear_ops, preconditioner_override=None):
        if len(linear_ops) > 2:
            raise RuntimeError("An AddedDiagLinearOperator can only have two components")

        if isinstance(linear_ops[0], DiagLinearOperator) and not isinstance(linear_ops[1], LowRankRootLinearOperator):
            raise RuntimeError(
                "A LowRankRootAddedDiagLinearOperator can only be created with a LowRankLinearOperator base!"
            )
        elif isinstance(linear_ops[1], DiagLinearOperator) and not isinstance(linear_ops[0], LowRankRootLinearOperator):
            raise RuntimeError(
                "A LowRankRootAddedDiagLinearOperator can only be created with a LowRankLinearOperator base!"
            )

        super().__init__(*linear_ops, preconditioner_override=preconditioner_override)

    @property
    @cached(name="chol_cap_mat")
    def chol_cap_mat(self):
        A_inv = self._diag_tensor.inverse()  # This is fine since it's a DiagLinearOperator
        U = self._linear_op.root
        V = self._linear_op.root.mT
        C = ConstantDiagLinearOperator(torch.ones(*V.batch_shape, 1, device=V.device, dtype=V.dtype), V.shape[-2])

        cap_mat = to_dense(C + V.matmul(A_inv.matmul(U)))
        chol_cap_mat = psd_safe_cholesky(cap_mat)

        return chol_cap_mat

    def _mul_constant(
        self: Float[LinearOperator, "*batch M N"], other: Union[float, torch.Tensor]
    ) -> Float[LinearOperator, "*batch M N"]:
        # We have to over-ride this here for the case where the constant is negative
        if other > 0:
            res = super()._mul_constant(other)
        else:
            res = AddedDiagLinearOperator(self._linear_op._mul_constant(other), self._diag_tensor._mul_constant(other))
        return res

    def _preconditioner(self) -> Tuple[Optional[Callable], Optional[LinearOperator], Optional[torch.Tensor]]:
        return None, None, None

    def _solve(
        self: Float[LinearOperator, "... N N"],
        rhs: Float[torch.Tensor, "... N C"],
        preconditioner: Optional[Callable[[Float[torch.Tensor, "... N C"]], Float[torch.Tensor, "... N C"]]] = None,
        num_tridiag: Optional[int] = 0,
    ) -> Union[
        Float[torch.Tensor, "... N C"],
        Tuple[
            Float[torch.Tensor, "... N C"],
            Float[torch.Tensor, "..."],  # Note that in case of a tuple the second term size depends on num_tridiag
        ],
    ]:
        A_inv = self._diag_tensor.inverse()  # This is fine since it's a DiagLinearOperator
        U = self._linear_op.root
        V = self._linear_op.root.mT
        chol_cap_mat = self.chol_cap_mat

        res = V.matmul(A_inv.matmul(rhs))
        res = torch.cholesky_solve(res, chol_cap_mat)
        res = A_inv.matmul(U.matmul(res))

        solve = A_inv.matmul(rhs) - res

        return solve

    def _solve_preconditioner(self) -> Optional[Callable]:
        return None

    def _sum_batch(self, dim: int) -> LinearOperator:
        return SumBatchLinearOperator(self, dim)

    def _logdet(self):
        chol_cap_mat = self.chol_cap_mat
        logdet_cap_mat = 2 * torch.diagonal(chol_cap_mat, offset=0, dim1=-2, dim2=-1).log().sum(-1)
        logdet_A = self._diag_tensor.logdet()
        logdet_term = logdet_cap_mat + logdet_A

        return logdet_term

    def __add__(
        self: Float[LinearOperator, "... #M #N"],
        other: Union[Float[Tensor, "... #M #N"], Float[LinearOperator, "... #M #N"], float],
    ) -> Union[Float[LinearOperator, "... M N"], Float[Tensor, "... M N"]]:
        from linear_operator.operators.diag_linear_operator import DiagLinearOperator

        if isinstance(other, DiagLinearOperator):
            return self.__class__(self._linear_op, self._diag_tensor + other)
        else:
            return AddedDiagLinearOperator(self._linear_op + other, self._diag_tensor)

    def inv_quad_logdet(
        self: Float[LinearOperator, "*batch N N"],
        inv_quad_rhs: Optional[Union[Float[Tensor, "*batch N M"], Float[Tensor, "*batch N"]]] = None,
        logdet: Optional[bool] = False,
        reduce_inv_quad: Optional[bool] = True,
    ) -> Tuple[
        Optional[Union[Float[Tensor, "*batch M"], Float[Tensor, " *batch"], Float[Tensor, " 0"]]],
        Optional[Float[Tensor, "..."]],
    ]:
        if not self.is_square:
            raise RuntimeError(
                "inv_quad_logdet only operates on (batches of) square (positive semi-definite) LinearOperators. "
                "Got a {} of size {}.".format(self.__class__.__name__, self.size())
            )

        if inv_quad_rhs is not None:
            if self.dim() == 2 and inv_quad_rhs.dim() == 1:
                if self.shape[-1] != inv_quad_rhs.numel():
                    raise RuntimeError(
                        "LinearOperator (size={}) cannot be multiplied with right-hand-side Tensor (size={}).".format(
                            self.shape, inv_quad_rhs.shape
                        )
                    )
            elif self.dim() != inv_quad_rhs.dim():
                raise RuntimeError(
                    "LinearOperator (size={}) and right-hand-side Tensor (size={}) should have the same number "
                    "of dimensions.".format(self.shape, inv_quad_rhs.shape)
                )
            elif self.batch_shape != inv_quad_rhs.shape[:-2] or self.shape[-1] != inv_quad_rhs.shape[-2]:
                raise RuntimeError(
                    "LinearOperator (size={}) cannot be multiplied with right-hand-side Tensor (size={}).".format(
                        self.shape, inv_quad_rhs.shape
                    )
                )

        inv_quad_term, logdet_term = None, None

        if inv_quad_rhs is not None:
            self_inv_rhs = self._solve(inv_quad_rhs)
            inv_quad_term = (inv_quad_rhs * self_inv_rhs).sum(dim=-2)
            if reduce_inv_quad:
                inv_quad_term = inv_quad_term.sum(dim=-1)

        if logdet:
            logdet_term = self._logdet()

        return inv_quad_term, logdet_term

    def solve(
        self: Float[LinearOperator, "... N N"],
        right_tensor: Union[Float[Tensor, "... N P"], Float[Tensor, " N"]],
        left_tensor: Optional[Float[Tensor, "... O N"]] = None,
    ) -> Union[Float[Tensor, "... N P"], Float[Tensor, "... N"], Float[Tensor, "... O P"], Float[Tensor, "... O"]]:
        if not self.is_square:
            raise RuntimeError(
                "solve only operates on (batches of) square (positive semi-definite) LinearOperators. "
                "Got a {} of size {}.".format(self.__class__.__name__, self.size())
            )

        if self.dim() == 2 and right_tensor.dim() == 1:
            if self.shape[-1] != right_tensor.numel():
                raise RuntimeError(
                    "LinearOperator (size={}) cannot be multiplied with right-hand-side Tensor (size={}).".format(
                        self.shape, right_tensor.shape
                    )
                )

        squeeze_solve = False
        if right_tensor.ndimension() == 1:
            right_tensor = right_tensor.unsqueeze(-1)
            squeeze_solve = True

        solve = self._solve(right_tensor)
        if squeeze_solve:
            solve = solve.squeeze(-1)

        if left_tensor is not None:
            return left_tensor @ solve
        else:
            return solve
