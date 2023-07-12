#!/usr/bin/env python3

from __future__ import annotations

import os
from typing import List, Optional, Tuple, Union

import torch
from jaxtyping import Float
from torch import Tensor

from .. import settings
from ..utils.memoize import cached
from ._linear_operator import IndexType, LinearOperator
from .block_diag_linear_operator import BlockDiagLinearOperator
from .dense_linear_operator import DenseLinearOperator
from .triangular_linear_operator import TriangularLinearOperator


class _DiagLinearOperator(TriangularLinearOperator):
    """
    Diagonal linear operator (... x N x N).

    :param diag: Diagonal elements of LinearOperator.
    """

    def __init__(self, diag: Float[Tensor, "*#batch N"]):
        super(TriangularLinearOperator, self).__init__(diag)
        self._diag = diag

    def __add__(
        self: Float[LinearOperator, "... #M #N"],
        other: Union[Float[Tensor, "... #M #N"], Float[LinearOperator, "... #M #N"], float],
    ) -> Union[Float[LinearOperator, "... M N"], Float[Tensor, "... M N"]]:
        if isinstance(other, DiagLinearOperator):
            return self.add_diagonal(other._diag)
        from .added_diag_linear_operator import AddedDiagLinearOperator

        return AddedDiagLinearOperator(other, self)

    def _bilinear_derivative(self, left_vecs: Tensor, right_vecs: Tensor) -> Tuple[Optional[Tensor], ...]:
        # TODO: Use proper batching for input vectors (prepend to shape rather than append)
        if not self._diag.requires_grad:
            return (None,)

        res = left_vecs * right_vecs
        if res.ndimension() > self._diag.ndimension():
            res = res.sum(-1)
        return (res,)

    @cached(name="cholesky", ignore_args=True)
    def _cholesky(
        self: Float[LinearOperator, "*batch N N"], upper: Optional[bool] = False
    ) -> Float[LinearOperator, "*batch N N"]:
        return self.sqrt()

    def _cholesky_solve(
        self: Float[LinearOperator, "*batch N N"],
        rhs: Union[Float[LinearOperator, "*batch2 N M"], Float[Tensor, "*batch2 N M"]],
        upper: Optional[bool] = False,
    ) -> Union[Float[LinearOperator, "... N M"], Float[Tensor, "... N M"]]:
        return rhs / self._diag.unsqueeze(-1).pow(2)

    def _expand_batch(
        self: Float[LinearOperator, "... M N"], batch_shape: Union[torch.Size, List[int]]
    ) -> Float[LinearOperator, "... M N"]:
        return self.__class__(self._diag.expand(*batch_shape, self._diag.size(-1)))

    def _diagonal(self: Float[LinearOperator, "... M N"]) -> Float[torch.Tensor, "... N"]:
        return self._diag

    def _get_indices(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> torch.Tensor:
        res = self._diag[(*batch_indices, row_index)]
        # If row and col index don't agree, then we have off diagonal elements
        # Those should be zero'd out
        res = res * torch.eq(row_index, col_index).to(device=res.device, dtype=res.dtype)
        return res

    def _mul_constant(
        self: Float[LinearOperator, "*batch M N"], other: Union[float, torch.Tensor]
    ) -> Float[LinearOperator, "*batch M N"]:
        return self.__class__(self._diag * other.unsqueeze(-1))

    def _mul_matrix(
        self: Float[LinearOperator, "... #M #N"],
        other: Union[Float[torch.Tensor, "... #M #N"], Float[LinearOperator, "... #M #N"]],
    ) -> Float[LinearOperator, "... M N"]:
        return DiagLinearOperator(self._diag * other._diagonal())

    def _prod_batch(self, dim: int) -> LinearOperator:
        return self.__class__(self._diag.prod(dim))

    def _root_decomposition(
        self: Float[LinearOperator, "... N N"]
    ) -> Union[Float[torch.Tensor, "... N N"], Float[LinearOperator, "... N N"]]:
        return self.sqrt()

    def _root_inv_decomposition(
        self: Float[LinearOperator, "*batch N N"],
        initial_vectors: Optional[torch.Tensor] = None,
        test_vectors: Optional[torch.Tensor] = None,
    ) -> Union[Float[LinearOperator, "... N N"], Float[Tensor, "... N N"]]:
        return self.inverse().sqrt()

    def _size(self) -> torch.Size:
        return torch.Size([*self._diag.shape, *self._diag.shape[-1:]])

    def _sum_batch(self, dim: int) -> LinearOperator:
        return self.__class__(self._diag.sum(dim))

    def _t_matmul(
        self: Float[LinearOperator, "*batch M N"],
        rhs: Union[Float[Tensor, "*batch2 M P"], Float[LinearOperator, "*batch2 M P"]],
    ) -> Union[Float[LinearOperator, "... N P"], Float[Tensor, "... N P"]]:
        # Diagonal matrices always commute
        return self._matmul(rhs)

    def _transpose_nonbatch(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch N M"]:
        return self

    def abs(self) -> LinearOperator:
        """
        Returns a DiagLinearOperator with the absolute value of all diagonal entries.
        """
        return self.__class__(self._diag.abs())

    def add_diagonal(
        self: Float[LinearOperator, "*batch N N"],
        diag: Union[Float[torch.Tensor, "... N"], Float[torch.Tensor, "... 1"], Float[torch.Tensor, ""]],
    ) -> Float[LinearOperator, "*batch N N"]:
        shape = torch.broadcast_shapes(self._diag.shape, diag.shape)
        return DiagLinearOperator(self._diag.expand(shape) + diag.expand(shape))

    @cached
    def to_dense(self: Float[LinearOperator, "*batch M N"]) -> Float[Tensor, "*batch M N"]:
        if self._diag.dim() == 0:
            return self._diag
        return torch.diag_embed(self._diag)

    def exp(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch M N"]:
        """
        Returns a DiagLinearOperator with all diagonal entries exponentiated.
        """
        return self.__class__(self._diag.exp())

    def inverse(self: Float[LinearOperator, "*batch N N"]) -> Float[LinearOperator, "*batch N N"]:
        """
        Returns the inverse of the DiagLinearOperator.
        """
        return self.__class__(self._diag.reciprocal())

    def inv_quad_logdet(
        self: Float[LinearOperator, "*batch N N"],
        inv_quad_rhs: Optional[Union[Float[Tensor, "*batch N M"], Float[Tensor, "*batch N"]]] = None,
        logdet: Optional[bool] = False,
        reduce_inv_quad: Optional[bool] = True,
    ) -> Tuple[
        Optional[Union[Float[Tensor, "*batch M"], Float[Tensor, " *batch"], Float[Tensor, " 0"]]],
        Optional[Float[Tensor, "..."]],
    ]:
        # TODO: Use proper batching for inv_quad_rhs (prepand to shape rathern than append)
        if inv_quad_rhs is None:
            rhs_batch_shape = torch.Size()
        else:
            rhs_batch_shape = inv_quad_rhs.shape[1 + self.batch_dim :]

        if inv_quad_rhs is None:
            inv_quad_term = torch.empty(0, dtype=self.dtype, device=self.device)
        else:
            diag = self._diag
            for _ in rhs_batch_shape:
                diag = diag.unsqueeze(-1)
            inv_quad_term = inv_quad_rhs.div(diag).mul(inv_quad_rhs).sum(-(1 + len(rhs_batch_shape)))
            if reduce_inv_quad:
                inv_quad_term = inv_quad_term.sum(-1)

        if not logdet:
            logdet_term = torch.empty(0, dtype=self.dtype, device=self.device)
        else:
            logdet_term = self._diag.log().sum(-1)

        return inv_quad_term, logdet_term

    def log(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch M N"]:
        """
        Returns a DiagLinearOperator with the log of all diagonal entries.
        """
        return self.__class__(self._diag.log())

    # this needs to be the public "matmul", instead of "_matmul", to hit the special cases before
    # a MatmulLinearOperator is created.
    def matmul(
        self: Float[LinearOperator, "*batch M N"],
        other: Union[Float[Tensor, "*batch2 N P"], Float[Tensor, "*batch2 N"], Float[LinearOperator, "*batch2 N P"]],
    ) -> Union[Float[Tensor, "... M P"], Float[Tensor, "... M"], Float[LinearOperator, "... M P"]]:
        if isinstance(other, Tensor):
            diag = self._diag if other.ndim == 1 else self._diag.unsqueeze(-1)
            return diag * other

        if isinstance(other, DenseLinearOperator):
            return DenseLinearOperator(self @ other.tensor)

        if isinstance(other, DiagLinearOperator):
            return DiagLinearOperator(self._diag * other._diag)

        if isinstance(other, TriangularLinearOperator):
            return TriangularLinearOperator(self @ other._tensor, upper=other.upper)

        if isinstance(other, BlockDiagLinearOperator):
            diag_reshape = self._diag.view(*other.base_linear_op.shape[:-1])
            diag = DiagLinearOperator(diag_reshape)
            # using matmul here avoids having to implement special case of elementwise multiplication
            # with block diagonal operator, which itself has special cases for vectors and matrices
            return BlockDiagLinearOperator(diag @ other.base_linear_op)

        return super().matmul(other)  # happens with other structured linear operators

    def _matmul(
        self: Float[LinearOperator, "*batch M N"],
        rhs: Union[Float[torch.Tensor, "*batch2 N C"], Float[torch.Tensor, "*batch2 N"]],
    ) -> Union[Float[torch.Tensor, "... M C"], Float[torch.Tensor, "... M"]]:
        return self.matmul(rhs)

    def solve(
        self: Float[LinearOperator, "... N N"],
        right_tensor: Union[Float[Tensor, "... N P"], Float[Tensor, " N"]],
        left_tensor: Optional[Float[Tensor, "... O N"]] = None,
    ) -> Union[Float[Tensor, "... N P"], Float[Tensor, "... N"], Float[Tensor, "... O P"], Float[Tensor, "... O"]]:
        res = self.inverse()._matmul(right_tensor)
        if left_tensor is not None:
            res = left_tensor @ res
        return res

    def solve_triangular(
        self, rhs: torch.Tensor, upper: bool, left: bool = True, unitriangular: bool = False
    ) -> torch.Tensor:
        # upper or lower doesn't matter here, it's all the same
        if unitriangular:
            if not torch.all(self.diagonal() == 1):
                raise RuntimeError("Received `unitriangular=True` but `LinearOperator` does not have a unit diagonal.")
            return rhs
        return self.solve(right_tensor=rhs)

    def sqrt(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch M N"]:
        """
        Returns a DiagLinearOperator with the square root of all diagonal entries.
        """
        return self.__class__(self._diag.sqrt())

    def sqrt_inv_matmul(
        self: Float[LinearOperator, "*batch N N"],
        rhs: Float[Tensor, "*batch N P"],
        lhs: Optional[Float[Tensor, "*batch O N"]] = None,
    ) -> Union[Float[Tensor, "*batch N P"], Tuple[Float[Tensor, "*batch O P"], Float[Tensor, "*batch O"]]]:
        matrix_inv_root = self._root_inv_decomposition()
        if lhs is None:
            return matrix_inv_root.matmul(rhs)
        else:
            sqrt_inv_matmul = lhs @ matrix_inv_root.matmul(rhs)
            inv_quad = (matrix_inv_root @ lhs.mT).mT.pow(2).sum(dim=-1)
            return sqrt_inv_matmul, inv_quad

    def zero_mean_mvn_samples(
        self: Float[LinearOperator, "*batch N N"], num_samples: int
    ) -> Float[Tensor, "num_samples *batch N"]:
        base_samples = torch.randn(num_samples, *self._diag.shape, dtype=self.dtype, device=self.device)
        return base_samples * self._diag.sqrt()

    @cached(name="svd")
    def _svd(
        self: Float[LinearOperator, "*batch N N"]
    ) -> Tuple[Float[LinearOperator, "*batch N N"], Float[Tensor, "... N"], Float[LinearOperator, "*batch N N"]]:
        evals, evecs = self._symeig(eigenvectors=True)
        S = torch.abs(evals)
        U = evecs
        V = evecs * torch.sign(evals).unsqueeze(-1)
        return U, S, V

    def _symeig(
        self: Float[LinearOperator, "*batch N N"],
        eigenvectors: bool = False,
        return_evals_as_lazy: Optional[bool] = False,
    ) -> Tuple[Float[Tensor, "*batch M"], Optional[Float[LinearOperator, "*batch N M"]]]:
        evals = self._diag
        if eigenvectors:
            diag_values = torch.ones(evals.shape[:-1], device=evals.device, dtype=evals.dtype).unsqueeze(-1)
            evecs = ConstantDiagLinearOperator(diag_values, diag_shape=evals.shape[-1])
        else:
            evecs = None
        return evals, evecs


if os.getenv("USE_COLA"):
    import cola
    from .cola_linear_operator import ColaLinearOperator

    class DiagLinearOperator(ColaLinearOperator):
        def _generate_cola_lo(self, diag):
            return cola.ops.Diagonal(diag)

        def _generate_orig_lo(self, diag):
            return _DiagLinearOperator(diag)

        @property
        def _diag(self):
            # This property is necessary for the evaluate_linear_operator
            # method of the test case
            return self._orig_lo._diag

else:
    DiagLinearOperator = _DiagLinearOperator


class ConstantDiagLinearOperator(DiagLinearOperator):
    """
    Diagonal lazy tensor with constant entries. Supports arbitrary batch sizes.
    Used e.g. for adding jitter to matrices.

    :param diag_values: A `... 1` Tensor, representing a
        of (batch of) `diag_shape x diag_shape` diagonal matrix.
    :param diag_shape: The (non-batch) dimension of the (square) matrix
    """

    def __init__(self, diag_values: torch.Tensor, diag_shape: int):
        if settings.debug.on():
            if not (diag_values.dim() and diag_values.size(-1) == 1):
                raise ValueError(
                    f"diag_values argument to ConstantDiagLinearOperator needs to have a final "
                    f"singleton dimension. Instead, got a value with shape {diag_values.shape}."
                )
        super(TriangularLinearOperator, self).__init__(diag_values, diag_shape=diag_shape)
        self.diag_values = diag_values
        self.diag_shape = diag_shape

    def __add__(
        self: Float[LinearOperator, "... #M #N"],
        other: Union[Float[Tensor, "... #M #N"], Float[LinearOperator, "... #M #N"], float],
    ) -> Union[Float[LinearOperator, "... M N"], Float[Tensor, "... M N"]]:
        if isinstance(other, ConstantDiagLinearOperator):
            if other.shape[-1] == self.shape[-1]:
                return ConstantDiagLinearOperator(self.diag_values + other.diag_values, self.diag_shape)
            raise RuntimeError(
                f"Trailing batch shapes must match for adding two ConstantDiagLinearOperators. "
                f"Instead, got shapes of {other.shape} and {self.shape}."
            )
        return super().__add__(other)

    def _bilinear_derivative(self, left_vecs: Tensor, right_vecs: Tensor) -> Tuple[Optional[Tensor], ...]:
        # TODO: Use proper batching for input vectors (prepand to shape rathern than append)
        if not self.diag_values.requires_grad:
            return (None,)

        res = (left_vecs * right_vecs).sum(dim=[-1, -2])
        res = res.unsqueeze(-1)
        return (res,)

    @property
    def _diag(self: Float[LinearOperator, "... N N"]) -> Float[Tensor, "... N"]:
        return self.diag_values.expand(*self.diag_values.shape[:-1], self.diag_shape)

    def _expand_batch(
        self: Float[LinearOperator, "... M N"], batch_shape: Union[torch.Size, List[int]]
    ) -> Float[LinearOperator, "... M N"]:
        return self.__class__(self.diag_values.expand(*batch_shape, 1), diag_shape=self.diag_shape)

    def _mul_constant(
        self: Float[LinearOperator, "*batch M N"], other: Union[float, torch.Tensor]
    ) -> Float[LinearOperator, "*batch M N"]:
        return self.__class__(self.diag_values * other, diag_shape=self.diag_shape)

    def _mul_matrix(
        self: Float[LinearOperator, "... #M #N"],
        other: Union[Float[torch.Tensor, "... #M #N"], Float[LinearOperator, "... #M #N"]],
    ) -> Float[LinearOperator, "... M N"]:
        if isinstance(other, ConstantDiagLinearOperator):
            if not self.diag_shape == other.diag_shape:
                raise ValueError(
                    "Dimension Mismatch: Must have same diag_shape, but got "
                    f"{self.diag_shape} and {other.diag_shape}"
                )
            return self.__class__(self.diag_values * other.diag_values, diag_shape=self.diag_shape)
        return super()._mul_matrix(other)

    def _prod_batch(self, dim: int) -> LinearOperator:
        return self.__class__(self.diag_values.prod(dim), diag_shape=self.diag_shape)

    def _size(self) -> torch.Size:
        # Though the super._size method works, this is more efficient
        return torch.Size([*self.diag_values.shape[:-1], self.diag_shape, self.diag_shape])

    def _sum_batch(self, dim: int) -> LinearOperator:
        return ConstantDiagLinearOperator(self.diag_values.sum(dim), diag_shape=self.diag_shape)

    def abs(self) -> LinearOperator:
        """
        Returns a DiagLinearOperator with the absolute value of all diagonal entries.
        """
        return ConstantDiagLinearOperator(self.diag_values.abs(), diag_shape=self.diag_shape)

    def exp(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch M N"]:
        """
        Returns a DiagLinearOperator with all diagonal entries exponentiated.
        """
        return ConstantDiagLinearOperator(self.diag_values.exp(), diag_shape=self.diag_shape)

    def inverse(self: Float[LinearOperator, "*batch N N"]) -> Float[LinearOperator, "*batch N N"]:
        """
        Returns the inverse of the DiagLinearOperator.
        """
        return ConstantDiagLinearOperator(self.diag_values.reciprocal(), diag_shape=self.diag_shape)

    def log(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch M N"]:
        """
        Returns a DiagLinearOperator with the log of all diagonal entries.
        """
        return ConstantDiagLinearOperator(self.diag_values.log(), diag_shape=self.diag_shape)

    def matmul(
        self: Float[LinearOperator, "*batch M N"],
        other: Union[Float[Tensor, "*batch2 N P"], Float[Tensor, "*batch2 N"], Float[LinearOperator, "*batch2 N P"]],
    ) -> Union[Float[Tensor, "... M P"], Float[Tensor, "... M"], Float[LinearOperator, "... M P"]]:
        if isinstance(other, ConstantDiagLinearOperator):
            return self._mul_matrix(other)
        return super().matmul(other)

    def solve_triangular(
        self, rhs: torch.Tensor, upper: bool, left: bool = True, unitriangular: bool = False
    ) -> torch.Tensor:
        return rhs / self.diag_values

    def sqrt(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch M N"]:
        """
        Returns a DiagLinearOperator with the square root of all diagonal entries.
        """
        return ConstantDiagLinearOperator(self.diag_values.sqrt(), diag_shape=self.diag_shape)
