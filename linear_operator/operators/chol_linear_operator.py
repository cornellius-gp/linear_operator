#!/usr/bin/env python3

from __future__ import annotations

import warnings
from typing import Callable, Optional, Tuple, Union

import torch
from jaxtyping import Float
from torch import Tensor

from ..utils.memoize import cached
from ._linear_operator import LinearOperator
from .root_linear_operator import RootLinearOperator
from .triangular_linear_operator import TriangularLinearOperator, _TriangularLinearOperatorBase


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

    def __init__(self, chol: Float[_TriangularLinearOperatorBase, "*#batch N N"], upper: bool = False):
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
    def _chol_diag(self: Float[LinearOperator, "*batch N N"]) -> Float[torch.Tensor, "... N"]:
        return self.root._diagonal()

    @cached(name="cholesky")
    def _cholesky(
        self: Float[LinearOperator, "*batch N N"], upper: Optional[bool] = False
    ) -> Float[LinearOperator, "*batch N N"]:
        if upper == self.upper:
            return self.root
        else:
            return self.root._transpose_nonbatch()

    @cached
    def _diagonal(self: Float[LinearOperator, "..."]) -> Float[torch.Tensor, "..."]:
        # TODO: Can we be smarter here?
        return (self.root.to_dense() ** 2).sum(-1)

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
        if num_tridiag:
            return super()._solve(rhs, preconditioner, num_tridiag=num_tridiag)
        return self.root._cholesky_solve(rhs, upper=self.upper)

    @cached
    def to_dense(self: Float[LinearOperator, "*batch M N"]) -> Float[Tensor, "*batch M N"]:
        root = self.root
        if self.upper:
            res = root._transpose_nonbatch() @ root
        else:
            res = root @ root._transpose_nonbatch()
        return res.to_dense()

    @cached
    def inverse(self: Float[LinearOperator, "*batch N N"]) -> Float[LinearOperator, "*batch N N"]:
        """
        Returns the inverse of the CholLinearOperator.
        """
        Linv = self.root.inverse()  # this could be slow in some cases w/ structured lazies
        return CholLinearOperator(TriangularLinearOperator(Linv, upper=not self.upper), upper=not self.upper)

    def inv_quad(
        self: Float[LinearOperator, "*batch N N"],
        inv_quad_rhs: Union[Float[Tensor, "*batch N M"], Float[Tensor, "*batch N"]],
        reduce_inv_quad: bool = True,
    ) -> Union[Float[Tensor, "*batch M"], Float[Tensor, " *batch"]]:
        if self.upper:
            R = self.root._transpose_nonbatch().solve(inv_quad_rhs)
        else:
            R = self.root.solve(inv_quad_rhs)
        inv_quad_term = (R**2).sum(dim=-2)
        if inv_quad_term.numel() and reduce_inv_quad:
            inv_quad_term = inv_quad_term.sum(-1)
        return inv_quad_term

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
        self: Float[LinearOperator, "*batch N N"],
        initial_vectors: Optional[torch.Tensor] = None,
        test_vectors: Optional[torch.Tensor] = None,
        method: Optional[str] = None,
    ) -> Union[Float[LinearOperator, "... N N"], Float[Tensor, "... N N"]]:
        inv_root = self.root.inverse()
        return RootLinearOperator(inv_root._transpose_nonbatch())

    def solve(
        self: Float[LinearOperator, "... N N"],
        right_tensor: Union[Float[Tensor, "... N P"], Float[Tensor, " N"]],
        left_tensor: Optional[Float[Tensor, "... O N"]] = None,
    ) -> Union[Float[Tensor, "... N P"], Float[Tensor, "... N"], Float[Tensor, "... O P"], Float[Tensor, "... O"]]:
        is_vector = right_tensor.ndim == 1
        if is_vector:
            right_tensor = right_tensor.unsqueeze(-1)
        res = self.root._cholesky_solve(right_tensor, upper=self.upper)
        if is_vector:
            res = res.squeeze(-1)
        if left_tensor is not None:
            res = left_tensor @ res
        return res
