#!/usr/bin/env python3

from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor

from ..utils.broadcasting import _mul_broadcast_shape
from ..utils.errors import NotPSDError
from ..utils.memoize import cached
from ._linear_operator import LinearOperator
from .batch_repeat_linear_operator import BatchRepeatLinearOperator
from .dense_linear_operator import DenseLinearOperator

Allsor = Union[Tensor, LinearOperator]


class _TriangularLinearOperatorBase:
    """Base class that all triangular lazy tensors are derived from."""

    pass


class TriangularLinearOperator(LinearOperator, _TriangularLinearOperatorBase):
    def __init__(self, tensor: Allsor, upper: bool = False) -> None:
        """
        Triangular lazy tensor. Supports arbitrary batch sizes.

        Args:
            :attr:`tensor` (Tensor or LinearOperator):
                A `b1 x ... x bk x n x n` Tensor, representing a `b1 x ... x bk`-sized batch
                of `n x n` triangular matrices.
            :attr:`upper` (bool):
                If True, the tensor is considered to be upper-triangular, otherwise lower-triangular.
        """
        if isinstance(tensor, TriangularLinearOperator):
            # this is a null-op, we can just use underlying tensor directly.
            tensor = tensor._tensor
        elif isinstance(tensor, BatchRepeatLinearOperator):
            # things get kind of messy when interleaving repeats and triangualrisms
            if not isinstance(tensor.base_linear_op, TriangularLinearOperator):
                tensor = tensor.__class__(
                    TriangularLinearOperator(tensor.base_linear_op, upper=upper),
                    batch_repeat=tensor.batch_repeat,
                )
        if torch.is_tensor(tensor):
            tensor = DenseLinearOperator(tensor)
        super().__init__(tensor)
        self.upper = upper
        self._tensor = tensor

    def __add__(self, other: Allsor) -> LinearOperator:
        from .diag_linear_operator import DiagLinearOperator

        if isinstance(other, DiagLinearOperator):
            from .added_diag_linear_operator import AddedDiagLinearOperator

            return self.__class__(AddedDiagLinearOperator(self._tensor, other), upper=self.upper)
        if isinstance(other, TriangularLinearOperator) and not self.upper ^ other.upper:
            return self.__class__(self._tensor + other._tensor, upper=self.upper)
        return self._tensor + other

    def _cholesky(self, upper=False) -> LinearOperator:
        raise NotPSDError("TriangularLinearOperator does not allow a Cholesky decomposition")

    def _cholesky_solve(self, rhs: Tensor, upper: bool = False) -> Tensor:
        # use custom method if implemented
        try:
            res = self._tensor._cholesky_solve(rhs=rhs, upper=upper)
        except NotImplementedError:
            if upper:
                # res = (U.T @ U)^-1 @ v = U^-1 @ U^-T @ v
                w = self._transpose_nonbatch().inv_matmul(rhs)
                res = self.inv_matmul(w)
            else:
                # res = (L @ L.T)^-1 @ v = L^-T @ L^-1 @ v
                w = self.inv_matmul(rhs)
                res = self._transpose_nonbatch().inv_matmul(w)
        return res

    def _diagonal(self) -> Tensor:
        return self._tensor._diagonal()

    def _get_indices(self, row_index, col_index, *batch_indices):
        return self._tensor._get_indices(row_index, col_index, *batch_indices)

    def _matmul(self, rhs: Tensor) -> Tensor:
        return self._tensor.matmul(rhs)

    def _mul_constant(self, constant: Tensor) -> "TriangularLinearOperator":
        return TriangularLinearOperator(self._tensor * constant.unsqueeze(-1), upper=self.upper)

    def _root_decomposition(self) -> Allsor:
        raise NotPSDError("TriangularLinearOperator does not allow a root decomposition")

    def _root_inv_decomposition(self, initial_vectors: Optional[Tensor] = None) -> Allsor:
        raise NotPSDError("TriangularLinearOperator does not allow an inverse root decomposition")

    def _size(self) -> torch.Size:
        return self._tensor.shape

    def _solve(
        self,
        rhs: Tensor,
        preconditioner: Callable[[Tensor], Tensor],
        num_tridiag: int = 0,
    ) -> Tensor:
        # already triangular, can just call inv_matmul for the solve
        return self.inv_matmul(rhs)

    def _sum_batch(self, dim: int) -> "TriangularLinearOperator":
        return TriangularLinearOperator(self._tensor._sum_batch(dim), upper=self.upper)

    def _transpose_nonbatch(self) -> "TriangularLinearOperator":
        return TriangularLinearOperator(self._tensor._transpose_nonbatch(), upper=not self.upper)

    def abs(self) -> "TriangularLinearOperator":
        return TriangularLinearOperator(self._tensor.abs(), upper=self.upper)

    def add_diagonal(self, added_diag: Tensor) -> "TriangularLinearOperator":
        from .added_diag_linear_operator import AddedDiagLinearOperator

        shape = _mul_broadcast_shape(self._diag.shape, added_diag.shape)
        added_diag_lt = AddedDiagLinearOperator(self._tensor.expand(shape), added_diag.expand(shape))
        return TriangularLinearOperator(added_diag_lt, upper=self.upper)

    @cached
    def to_dense(self) -> Tensor:
        return self._tensor.to_dense()

    def exp(self) -> "TriangularLinearOperator":
        return TriangularLinearOperator(self._tensor.exp(), upper=self.upper)

    def inv_matmul(self, right_tensor: Tensor, left_tensor: Optional[Tensor] = None) -> Tensor:
        if isinstance(self._tensor, DenseLinearOperator):
            res = torch.triangular_solve(right_tensor, self.to_dense(), upper=self.upper).solution
        elif isinstance(self._tensor, BatchRepeatLinearOperator):
            res = self._tensor.base_linear_op.inv_matmul(right_tensor, left_tensor)
            # TODO: Proper broadcasting
            res = res.expand(self._tensor.batch_repeat + res.shape[-2:])
        else:
            # TODO: Can we be smarter here?
            res = self._tensor.inv_matmul(right_tensor=right_tensor, left_tensor=left_tensor)
        if left_tensor is not None:
            res = left_tensor @ res
        return res

    def inv_quad_logdet(
        self,
        inv_quad_rhs: Optional[Tensor] = None,
        logdet: bool = False,
        reduce_inv_quad: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        if inv_quad_rhs is None:
            inv_quad_term = torch.empty(0, dtype=self.dtype, device=self.device)
        else:
            # triangular, inv_matmul is cheap
            inv_quad_term = inv_quad_rhs.transpose(-1, -2) @ self.inv_matmul(inv_quad_rhs)
        if logdet:
            diag = self._diagonal()
            logdet_term = self._diagonal().abs().log().sum(-1)
            if torch.sign(diag).prod(-1) < 0:
                logdet_term = torch.full_like(logdet_term, float("nan"))
        else:
            logdet_term = torch.empty(0, dtype=self.dtype, device=self.device)
        if inv_quad_term.numel() and reduce_inv_quad:
            inv_quad_term = inv_quad_term.sum(-1)
        return inv_quad_term, logdet_term

    @cached
    def inverse(self) -> "TriangularLinearOperator":
        eye = torch.eye(self._tensor.size(-1), device=self._tensor.device, dtype=self._tensor.dtype)
        inv = self.inv_matmul(eye)
        return TriangularLinearOperator(inv, upper=self.upper)

    def _expand_batch(self, batch_shape):
        if len(batch_shape) == 0:
            return self
        return self.__class__(tensor=self._tensor._expand_batch(batch_shape), upper=self.upper)
