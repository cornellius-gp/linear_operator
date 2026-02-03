#!/usr/bin/env python3

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from linear_operator import settings
from linear_operator.operators._linear_operator import IndexType, LinearOperator
from linear_operator.operators.block_diag_linear_operator import BlockDiagLinearOperator
from linear_operator.operators.dense_linear_operator import DenseLinearOperator
from linear_operator.operators.triangular_linear_operator import TriangularLinearOperator
from linear_operator.utils.memoize import cached


class DiagLinearOperator(TriangularLinearOperator):
    """
    Diagonal linear operator (... x N x N).

    :param diag: Diagonal elements of LinearOperator.
    """

    def __init__(self, diag: Tensor):
        super(TriangularLinearOperator, self).__init__(diag)
        self._diag = diag

    def __add__(
        self: LinearOperator,  # shape: (..., #M, #N)
        other: Union[Tensor, LinearOperator, float],  # shape: (..., #M, #N)
    ) -> Union[LinearOperator, Tensor]:  # shape: (..., M, N)
        if isinstance(other, DiagLinearOperator):
            return self.add_diagonal(other._diag)
        from linear_operator.operators.added_diag_linear_operator import AddedDiagLinearOperator

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
        self: LinearOperator, upper: Optional[bool] = False  # shape: (*batch, N, N)
    ) -> LinearOperator:  # shape: (*batch, N, N)
        return self.sqrt()

    def _cholesky_solve(
        self: LinearOperator,  # shape: (*batch, N, N)
        rhs: Union[LinearOperator, Tensor],  # shape: (*batch2, N, M)
        upper: Optional[bool] = False,
    ) -> Union[LinearOperator, Tensor]:  # shape: (..., N, M)
        return rhs / self._diag.unsqueeze(-1).pow(2)

    def _expand_batch(
        self: LinearOperator, batch_shape: Union[torch.Size, List[int]]  # shape: (..., M, N)
    ) -> LinearOperator:  # shape: (..., M, N)
        return self.__class__(self._diag.expand(*batch_shape, self._diag.size(-1)))

    def _diagonal(
        self: LinearOperator,  # shape: (..., M, N)
    ) -> torch.Tensor:  # shape: (..., N)
        return self._diag

    def _get_indices(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> torch.Tensor:
        res = self._diag[(*batch_indices, row_index)]
        # Unify device and dtype prior to comparison
        row_index = row_index.to(device=res.device, dtype=res.dtype)
        col_index = col_index.to(device=res.device, dtype=res.dtype)
        # If row and col index don't agree, then we have off diagonal elements
        # Those should be zero'd out
        res = res * torch.eq(row_index, col_index)
        return res

    def _mul_constant(
        self: LinearOperator, other: Union[float, torch.Tensor]  # shape: (*batch, M, N)
    ) -> LinearOperator:  # shape: (*batch, M, N)
        return self.__class__(self._diag * other.unsqueeze(-1))

    def _mul_matrix(
        self: LinearOperator,  # shape: (..., #M, #N)
        other: Union[torch.Tensor, LinearOperator],  # shape: (..., #M, #N)
    ) -> LinearOperator:  # shape: (..., M, N)
        return DiagLinearOperator(self._diag * other._diagonal())

    def _prod_batch(self, dim: int) -> LinearOperator:
        return self.__class__(self._diag.prod(dim))

    def _root_decomposition(
        self: LinearOperator,  # shape: (..., N, N)
    ) -> Union[torch.Tensor, LinearOperator]:  # shape: (..., N, N)
        return self.sqrt()

    def _root_inv_decomposition(
        self: LinearOperator,  # shape: (*batch, N, N)
        initial_vectors: Optional[torch.Tensor] = None,
        test_vectors: Optional[torch.Tensor] = None,
    ) -> Union[LinearOperator, Tensor]:  # shape: (..., N, N)
        return self.inverse().sqrt()

    def _size(self) -> torch.Size:
        return torch.Size([*self._diag.shape, *self._diag.shape[-1:]])

    def _sum_batch(self, dim: int) -> LinearOperator:
        return self.__class__(self._diag.sum(dim))

    def _t_matmul(
        self: LinearOperator,  # shape: (*batch, M, N)
        rhs: Union[Tensor, LinearOperator],  # shape: (*batch2, M, P)
    ) -> Union[LinearOperator, Tensor]:  # shape: (..., N, P)
        # Diagonal matrices always commute
        return self._matmul(rhs)

    def _transpose_nonbatch(
        self: LinearOperator,  # shape: (*batch, M, N)
    ) -> LinearOperator:  # shape: (*batch, N, M)
        return self

    def abs(self) -> LinearOperator:
        """
        Returns a DiagLinearOperator with the absolute value of all diagonal entries.
        """
        return self.__class__(self._diag.abs())

    def add_diagonal(
        self: LinearOperator,  # shape: (*batch, N, N)
        diag: torch.Tensor,  # shape: (..., N) or (..., 1) or ()
    ) -> LinearOperator:  # shape: (*batch, N, N)
        shape = torch.broadcast_shapes(self._diag.shape, diag.shape)
        return DiagLinearOperator(self._diag.expand(shape) + diag.expand(shape))

    @cached
    def to_dense(
        self: LinearOperator,  # shape: (*batch, M, N)
    ) -> Tensor:  # shape: (*batch, M, N)
        if self._diag.dim() == 0:
            return self._diag
        return torch.diag_embed(self._diag)

    def exp(
        self: LinearOperator,  # shape: (*batch, M, N)
    ) -> LinearOperator:  # shape: (*batch, M, N)
        """
        Returns a DiagLinearOperator with all diagonal entries exponentiated.
        """
        return self.__class__(self._diag.exp())

    def inverse(
        self: LinearOperator,  # shape: (*batch, N, N)
    ) -> LinearOperator:  # shape: (*batch, N, N)
        """
        Returns the inverse of the DiagLinearOperator.
        """
        return self.__class__(self._diag.reciprocal())

    def inv_quad_logdet(
        self: LinearOperator,  # shape: (*batch, N, N)
        inv_quad_rhs: Optional[Tensor] = None,  # shape: (*batch, N, M) or (*batch, N)
        logdet: Optional[bool] = False,
        reduce_inv_quad: Optional[bool] = True,
    ) -> Tuple[  # fmt: off
        Optional[Tensor],  # shape: (*batch, M) or (*batch) or (0)
        Optional[Tensor],  # shape: (...)
    ]:  # fmt: on
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

    def log(
        self: LinearOperator,  # shape: (*batch, M, N)
    ) -> LinearOperator:  # shape: (*batch, M, N)
        """
        Returns a DiagLinearOperator with the log of all diagonal entries.
        """
        return self.__class__(self._diag.log())

    # this needs to be the public "matmul", instead of "_matmul", to hit the special cases before
    # a MatmulLinearOperator is created.
    def matmul(
        self: LinearOperator,  # shape: (*batch, M, N)
        other: Union[Tensor, LinearOperator],  # shape: (*batch2, N, P) or (*batch2, N)
    ) -> Union[Tensor, LinearOperator]:  # shape: (..., M, P) or (..., M)
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
        self: LinearOperator,  # shape: (*batch, M, N)
        rhs: torch.Tensor,  # shape: (*batch2, N, C) or (*batch2, N)
    ) -> torch.Tensor:  # shape: (..., M, C) or (..., M)
        return self.matmul(rhs)

    def solve(
        self: LinearOperator,  # shape: (..., N, N)
        right_tensor: Tensor,  # shape: (..., N, P) or (N)
        left_tensor: Optional[Tensor] = None,  # shape: (..., O, N)
    ) -> Tensor:  # shape: (..., N, P) or (..., N) or (..., O, P) or (..., O)
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

    def sqrt(
        self: LinearOperator,  # shape: (*batch, M, N)
    ) -> LinearOperator:  # shape: (*batch, M, N)
        """
        Returns a DiagLinearOperator with the square root of all diagonal entries.
        """
        return self.__class__(self._diag.sqrt())

    def sqrt_inv_matmul(
        self: LinearOperator,  # shape: (*batch, N, N)
        rhs: Tensor,  # shape: (*batch, N, P)
        lhs: Optional[Tensor] = None,  # shape: (*batch, O, N)
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:  # shape: (*batch, N, P), (*batch, O, P), (*batch, O)
        matrix_inv_root = self._root_inv_decomposition()
        if lhs is None:
            return matrix_inv_root.matmul(rhs)
        else:
            sqrt_inv_matmul = lhs @ matrix_inv_root.matmul(rhs)
            inv_quad = (matrix_inv_root @ lhs.mT).mT.pow(2).sum(dim=-1)
            return sqrt_inv_matmul, inv_quad

    def zero_mean_mvn_samples(
        self: LinearOperator, num_samples: int  # shape: (*batch, N, N)
    ) -> Tensor:  # shape: (num_samples, *batch, N)
        base_samples = torch.randn(num_samples, *self._diag.shape, dtype=self.dtype, device=self.device)
        return base_samples * self._diag.sqrt()

    @cached(name="svd")
    def _svd(
        self: LinearOperator,  # shape: (*batch, N, N)
    ) -> Tuple[LinearOperator, Tensor, LinearOperator]:  # shape: (*batch, N, N), (..., N), (*batch, N, N)
        evals, evecs = self._symeig(eigenvectors=True)
        S = torch.abs(evals)
        U = evecs
        V = evecs * torch.sign(evals).unsqueeze(-1)
        return U, S, V

    def _symeig(
        self: LinearOperator,  # shape: (*batch, N, N)
        eigenvectors: bool = False,
        return_evals_as_lazy: Optional[bool] = False,
    ) -> Tuple[Tensor, Optional[LinearOperator]]:  # shape: (*batch, M), (*batch, N, M)
        evals = self._diag
        if eigenvectors:
            diag_values = torch.ones(evals.shape[:-1], device=evals.device, dtype=evals.dtype).unsqueeze(-1)
            evecs = ConstantDiagLinearOperator(diag_values, diag_shape=evals.shape[-1])
        else:
            evecs = None
        return evals, evecs


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
        self: LinearOperator,  # shape: (..., #M, #N)
        other: Union[Tensor, LinearOperator, float],  # shape: (..., #M, #N)
    ) -> Union[LinearOperator, Tensor]:  # shape: (..., M, N)
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
    def _diag(
        self: LinearOperator,  # shape: (..., N, N)
    ) -> Tensor:  # shape: (..., N)
        return self.diag_values.expand(*self.diag_values.shape[:-1], self.diag_shape)

    def _expand_batch(
        self: LinearOperator, batch_shape: Union[torch.Size, List[int]]  # shape: (..., M, N)
    ) -> LinearOperator:  # shape: (..., M, N)
        return self.__class__(self.diag_values.expand(*batch_shape, 1), diag_shape=self.diag_shape)

    def _mul_constant(
        self: LinearOperator, other: Union[float, torch.Tensor]  # shape: (*batch, M, N)
    ) -> LinearOperator:  # shape: (*batch, M, N)
        return self.__class__(self.diag_values * other, diag_shape=self.diag_shape)

    def _mul_matrix(
        self: LinearOperator,  # shape: (..., #M, #N)
        other: Union[torch.Tensor, LinearOperator],  # shape: (..., #M, #N)
    ) -> LinearOperator:  # shape: (..., M, N)
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

    def exp(
        self: LinearOperator,  # shape: (*batch, M, N)
    ) -> LinearOperator:  # shape: (*batch, M, N)
        """
        Returns a DiagLinearOperator with all diagonal entries exponentiated.
        """
        return ConstantDiagLinearOperator(self.diag_values.exp(), diag_shape=self.diag_shape)

    def inverse(
        self: LinearOperator,  # shape: (*batch, N, N)
    ) -> LinearOperator:  # shape: (*batch, N, N)
        """
        Returns the inverse of the DiagLinearOperator.
        """
        return ConstantDiagLinearOperator(self.diag_values.reciprocal(), diag_shape=self.diag_shape)

    def log(
        self: LinearOperator,  # shape: (*batch, M, N)
    ) -> LinearOperator:  # shape: (*batch, M, N)
        """
        Returns a DiagLinearOperator with the log of all diagonal entries.
        """
        return ConstantDiagLinearOperator(self.diag_values.log(), diag_shape=self.diag_shape)

    def matmul(
        self: LinearOperator,  # shape: (*batch, M, N)
        other: Union[Tensor, LinearOperator],  # shape: (*batch2, N, P) or (*batch2, N)
    ) -> Union[Tensor, LinearOperator]:  # shape: (..., M, P) or (..., M)
        if isinstance(other, ConstantDiagLinearOperator):
            return self._mul_matrix(other)
        return super().matmul(other)

    def solve_triangular(
        self, rhs: torch.Tensor, upper: bool, left: bool = True, unitriangular: bool = False
    ) -> torch.Tensor:
        return rhs / self.diag_values

    def sqrt(
        self: LinearOperator,  # shape: (*batch, M, N)
    ) -> LinearOperator:  # shape: (*batch, M, N)
        """
        Returns a DiagLinearOperator with the square root of all diagonal entries.
        """
        return ConstantDiagLinearOperator(self.diag_values.sqrt(), diag_shape=self.diag_shape)
