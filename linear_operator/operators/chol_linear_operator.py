#!/usr/bin/env python3

from __future__ import annotations

import warnings
from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor

from linear_operator.operators._linear_operator import LinearOperator
from linear_operator.operators.root_linear_operator import RootLinearOperator
from linear_operator.operators.triangular_linear_operator import _TriangularLinearOperatorBase, TriangularLinearOperator

from linear_operator.utils.memoize import cached


class CholLinearOperator(RootLinearOperator):
    r"""
    A LinearOperator (... x N x N) that represents a positive definite matrix given
    a lower trinagular Cholesky factor :math:`\mathbf L`
    (or upper triangular Cholesky factor :math:`\mathbf R`).

    :param chol: The Cholesky factor :math:`\mathbf L` (or :math:`\mathbf R`).
    :type chol: TriangularLinearOperator (... x N x N)
    :param upper: If the orientation of the cholesky factor is an upper triangular matrix
        (i.e. :math:`\mathbf R^\top \mathbf R`).
        If false, then the orientation is assumed to be a lower triangular matrix
        (i.e. :math:`\mathbf L \mathbf L^\top`).
    """

    def __init__(self, chol: _TriangularLinearOperatorBase, upper: bool = False):
        if not isinstance(chol, _TriangularLinearOperatorBase):
            warnings.warn(
                "chol argument to CholLinearOperator should be a TriangularLinearOperator. "
                "Passing a dense tensor will cause errors in future versions.",
                DeprecationWarning,
            )
            if torch.all(torch.tril(chol) == chol):
                chol = TriangularLinearOperator(chol, upper=False)
            elif torch.all(torch.triu(chol) == chol):
                chol = TriangularLinearOperator(chol, upper=True)
            else:
                raise ValueError("chol must be either lower or upper triangular")
        super().__init__(chol)
        self.upper = upper

    @property
    def _chol_diag(
        self: LinearOperator,  # shape: (*batch, N, N)
    ) -> torch.Tensor:  # shape: (..., N)
        return self.root._diagonal()

    @cached(name="cholesky")
    def _cholesky(
        self: LinearOperator, upper: Optional[bool] = False  # shape: (*batch, N, N)
    ) -> LinearOperator:  # shape: (*batch, N, N)
        if upper == self.upper:
            return self.root
        else:
            return self.root._transpose_nonbatch()

    @cached
    def _diagonal(
        self: LinearOperator,  # shape: (..., M, N)
    ) -> torch.Tensor:  # shape: (..., N)
        # TODO: Can we be smarter here?
        return (self.root.to_dense() ** 2).sum(-1)

    def _solve(
        self: LinearOperator,  # shape: (..., N, N)
        rhs: torch.Tensor,  # shape: (..., N, C)
        preconditioner: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,  # shape: (..., N, C)
        num_tridiag: Optional[int] = 0,
    ) -> Union[
        torch.Tensor,  # shape: (..., N, C)
        Tuple[
            torch.Tensor,  # shape: (..., N, C)
            torch.Tensor,  # Note that in case of a tuple the second term size depends on num_tridiag  # shape: (...)
        ],
    ]:
        if num_tridiag:
            return super()._solve(rhs, preconditioner, num_tridiag=num_tridiag)
        return self.root._cholesky_solve(rhs, upper=self.upper)

    @cached
    def to_dense(
        self: LinearOperator,  # shape: (*batch, M, N)
    ) -> Tensor:  # shape: (*batch, M, N)
        root = self.root
        if self.upper:
            res = root._transpose_nonbatch() @ root
        else:
            res = root @ root._transpose_nonbatch()
        return res.to_dense()

    @cached
    def inverse(
        self: LinearOperator,  # shape: (*batch, N, N)
    ) -> LinearOperator:  # shape: (*batch, N, N)
        """
        Returns the inverse of the CholLinearOperator.
        """
        Linv = self.root.inverse()  # this could be slow in some cases w/ structured lazies
        return CholLinearOperator(TriangularLinearOperator(Linv, upper=not self.upper), upper=not self.upper)

    def inv_quad(
        self: LinearOperator,  # shape: (*batch, N, N)
        inv_quad_rhs: Tensor,  # shape: (*batch, N, M) or (*batch, N)
        reduce_inv_quad: bool = True,
    ) -> Tensor:  # shape: (*batch, M) or (*batch)
        if self.upper:
            R = self.root._transpose_nonbatch().solve(inv_quad_rhs)
        else:
            R = self.root.solve(inv_quad_rhs)
        inv_quad_term = (R**2).sum(dim=-2)
        if inv_quad_term.numel() and reduce_inv_quad:
            inv_quad_term = inv_quad_term.sum(-1)
        return inv_quad_term

    def inv_quad_logdet(
        self: LinearOperator,  # shape: (*batch, N, N)
        inv_quad_rhs: Optional[Tensor] = None,  # shape: (*batch, N, M) or (*batch, N)
        logdet: Optional[bool] = False,
        reduce_inv_quad: Optional[bool] = True,
    ) -> Tuple[  # fmt: off
        Optional[Tensor],  # shape: (*batch, M) or (*batch) or (0)
        Optional[Tensor],  # shape: (...)
    ]:  # fmt: on
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
            elif self.shape[-1] != inv_quad_rhs.shape[-2]:
                raise RuntimeError(
                    "LinearOperator (size={}) cannot be multiplied with right-hand-side Tensor (size={}).".format(
                        self.shape, inv_quad_rhs.shape
                    )
                )

        inv_quad_term = None
        logdet_term = None

        if inv_quad_rhs is not None:
            inv_quad_term = self.inv_quad(inv_quad_rhs, reduce_inv_quad=reduce_inv_quad)

        if logdet:
            logdet_term = self._chol_diag.pow(2).log().sum(-1)

        return inv_quad_term, logdet_term

    def root_inv_decomposition(
        self: LinearOperator,  # shape: (*batch, N, N)
        initial_vectors: Optional[torch.Tensor] = None,
        test_vectors: Optional[torch.Tensor] = None,
        method: Optional[str] = None,
    ) -> Union[LinearOperator, Tensor]:  # shape: (..., N, N)
        inv_root = self.root.inverse()
        return RootLinearOperator(inv_root._transpose_nonbatch())

    def solve(
        self: LinearOperator,  # shape: (..., N, N)
        right_tensor: Tensor,  # shape: (..., N, P) or (N)
        left_tensor: Optional[Tensor] = None,  # shape: (..., O, N)
    ) -> Tensor:  # shape: (..., N, P) or (..., N) or (..., O, P) or (..., O)
        is_vector = right_tensor.ndim == 1
        if is_vector:
            right_tensor = right_tensor.unsqueeze(-1)
        res = self.root._cholesky_solve(right_tensor, upper=self.upper)
        if is_vector:
            res = res.squeeze(-1)
        if left_tensor is not None:
            res = left_tensor @ res
        return res
