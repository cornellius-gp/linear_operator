#!/usr/bin/env python3

from abc import ABCMeta
from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor

from linear_operator.operators._linear_operator import IndexType, LinearOperator
from linear_operator.operators.block_linear_operator import BlockLinearOperator
from linear_operator.operators.dense_linear_operator import DenseLinearOperator

from linear_operator.utils.memoize import cached


# metaclass of BlockDiagLinearOperator, overwrites behavior of constructor call
# _MetaBlockDiagLinearOperator(base_linear_op, block_dim=-3) to return a DiagLinearOperator
# if base_linear_op is a DiagLinearOperator itself
class _MetaBlockDiagLinearOperator(ABCMeta):
    def __call__(cls, base_linear_op: Union[LinearOperator, Tensor], block_dim: int = -3):
        from linear_operator.operators.diag_linear_operator import DiagLinearOperator

        if cls is BlockDiagLinearOperator and isinstance(base_linear_op, DiagLinearOperator):
            if block_dim != -3:
                raise NotImplementedError(
                    "Passing a base_linear_op of type DiagLinearOperator to the constructor of "
                    f"BlockDiagLinearOperator with block_dim = {block_dim} != -3 is not supported."
                )
            else:
                diag = base_linear_op._diag.flatten(-2, -1)  # flatten last two dimensions of diag
                return DiagLinearOperator(diag)
        else:
            return type.__call__(cls, base_linear_op, block_dim)


class BlockDiagLinearOperator(BlockLinearOperator, metaclass=_MetaBlockDiagLinearOperator):
    """
    Represents a lazy tensor that is the block diagonal of square matrices.
    The :attr:`block_dim` attribute specifies which dimension of the base LinearOperator
    specifies the blocks.
    For example, (with `block_dim=-3` a `k x n x n` tensor represents `k` `n x n` blocks (a `kn x kn` matrix).
    A `b x k x n x n` tensor represents `k` `b x n x n` blocks (a `b x kn x kn` batch matrix).

    Args:
        :attr:`base_linear_op` (LinearOperator or Tensor):
            Must be at least 3 dimensional.
        :attr:`block_dim` (int):
            The dimension that specifies the blocks.
    """

    def __init__(self, base_linear_op, block_dim=-3):
        if isinstance(base_linear_op, Tensor):
            base_linear_op = DenseLinearOperator(base_linear_op)

        super().__init__(base_linear_op, block_dim)
        # block diagonal is restricted to have square diagonal blocks
        if self.base_linear_op.shape[-1] != self.base_linear_op.shape[-2]:
            raise RuntimeError(
                "base_linear_op must be a batch of square matrices, but non-batch dimensions are "
                f"{base_linear_op.shape[-2:]}"
            )

    @property
    def num_blocks(self) -> int:
        return self.base_linear_op.size(-3)

    def _add_batch_dim(
        self: LinearOperator, other: torch.Tensor  # shape: (*batch1, M, P) or (*batch2, N, C)
    ) -> torch.Tensor:  # shape: (batch2, ..., C)
        *batch_shape, num_rows, num_cols = other.shape
        batch_shape = list(batch_shape)

        batch_shape.append(self.num_blocks)
        other = other.view(*batch_shape, num_rows // self.num_blocks, num_cols)
        return other

    @cached(name="cholesky")
    def _cholesky(
        self: LinearOperator, upper: Optional[bool] = False  # shape: (*batch, N, N)
    ) -> LinearOperator:  # shape: (*batch, N, N)
        from linear_operator.operators.triangular_linear_operator import TriangularLinearOperator

        chol = self.__class__(self.base_linear_op.cholesky(upper=upper))
        return TriangularLinearOperator(chol, upper=upper)

    def _cholesky_solve(
        self: LinearOperator,  # shape: (*batch, N, N)
        rhs: Union[LinearOperator, Tensor],  # shape: (*batch2, N, M)
        upper: Optional[bool] = False,
    ) -> Union[LinearOperator, Tensor]:  # shape: (..., N, M)
        rhs = self._add_batch_dim(rhs)
        res = self.base_linear_op._cholesky_solve(rhs, upper=upper)
        res = self._remove_batch_dim(res)
        return res

    def _diagonal(
        self: LinearOperator,  # shape: (..., M, N)
    ) -> torch.Tensor:  # shape: (..., N)
        res = self.base_linear_op._diagonal().contiguous()
        return res.view(*self.batch_shape, self.size(-1))

    def _get_indices(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> torch.Tensor:
        # Figure out what block the row/column indices belong to
        row_index_block = torch.div(row_index, self.base_linear_op.size(-2), rounding_mode="floor")
        col_index_block = torch.div(col_index, self.base_linear_op.size(-1), rounding_mode="floor")

        # Find the row/col index within each block
        row_index = row_index.fmod(self.base_linear_op.size(-2))
        col_index = col_index.fmod(self.base_linear_op.size(-1))

        # If the row/column blocks do not agree, then we have off diagonal elements
        # These elements should be zeroed out
        res = self.base_linear_op._get_indices(row_index, col_index, *batch_indices, row_index_block)
        res = res * torch.eq(row_index_block, col_index_block).type_as(res)
        return res

    def _remove_batch_dim(self, other):
        shape = list(other.shape)
        del shape[-3]
        shape[-2] *= self.num_blocks
        other = other.reshape(*shape)
        return other

    def _root_decomposition(
        self: LinearOperator,  # shape: (..., N, N)
    ) -> Union[torch.Tensor, LinearOperator]:  # shape: (..., N, N)
        return self.__class__(self.base_linear_op._root_decomposition())

    def _root_inv_decomposition(
        self: LinearOperator,  # shape: (*batch, N, N)
        initial_vectors: Optional[torch.Tensor] = None,
        test_vectors: Optional[torch.Tensor] = None,
    ) -> Union[LinearOperator, Tensor]:  # shape: (..., N, N)
        return self.__class__(self.base_linear_op._root_inv_decomposition(initial_vectors))

    def _size(self) -> torch.Size:
        shape = list(self.base_linear_op.shape)
        shape[-2] *= shape[-3]
        shape[-1] *= shape[-3]
        del shape[-3]
        return torch.Size(shape)

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
        else:
            rhs = self._add_batch_dim(rhs)
            res = self.base_linear_op._solve(rhs, preconditioner, num_tridiag=None)
            res = self._remove_batch_dim(res)
            return res

    def inv_quad_logdet(
        self: LinearOperator,  # shape: (*batch, N, N)
        inv_quad_rhs: Optional[Tensor] = None,  # shape: (*batch, N, M) or (*batch, N)
        logdet: Optional[bool] = False,
        reduce_inv_quad: Optional[bool] = True,
    ) -> Tuple[
        Optional[Tensor],  # shape: (*batch, M) or (*batch) or (0)
        Optional[Tensor],  # shape: (...)
    ]:
        if inv_quad_rhs is not None:
            inv_quad_rhs = self._add_batch_dim(inv_quad_rhs)
        inv_quad_res, logdet_res = self.base_linear_op.inv_quad_logdet(
            inv_quad_rhs, logdet, reduce_inv_quad=reduce_inv_quad
        )
        if inv_quad_res is not None and inv_quad_res.numel():
            if reduce_inv_quad:
                inv_quad_res = inv_quad_res.view(*self.base_linear_op.batch_shape)
                inv_quad_res = inv_quad_res.sum(-1)
            else:
                inv_quad_res = inv_quad_res.view(*self.base_linear_op.batch_shape, inv_quad_res.size(-1))
                inv_quad_res = inv_quad_res.sum(-2)
        if logdet_res is not None and logdet_res.numel():
            logdet_res = logdet_res.view(*logdet_res.shape).sum(-1)
        return inv_quad_res, logdet_res

    def matmul(
        self: LinearOperator,  # shape: (*batch, M, N)
        other: Union[Tensor, LinearOperator],  # shape: (*batch2, N, P) or (*batch2, N)
    ) -> Union[Tensor, LinearOperator]:  # shape: (..., M, P) or (..., M)
        from linear_operator.operators.diag_linear_operator import DiagLinearOperator

        # this is trivial if we multiply two BlockDiagLinearOperator with matching block sizes
        if isinstance(other, BlockDiagLinearOperator) and self.base_linear_op.shape == other.base_linear_op.shape:
            return BlockDiagLinearOperator(self.base_linear_op @ other.base_linear_op)
        # special case if we have a DiagLinearOperator
        if isinstance(other, DiagLinearOperator):
            # matmul is going to be cheap because of the special casing in DiagLinearOperator
            diag_reshape = other._diag.view(*self.base_linear_op.shape[:-1])
            diag = DiagLinearOperator(diag_reshape)
            return BlockDiagLinearOperator(self.base_linear_op @ diag)
        return super().matmul(other)

    @cached(name="svd")
    def _svd(
        self: LinearOperator,  # shape: (*batch, N, N)
    ) -> Tuple[LinearOperator, Tensor, LinearOperator]:  # shape: (*batch, N, N), (..., N), (*batch, N, N)
        U, S, V = self.base_linear_op.svd()
        # Doesn't make much sense to sort here, o/w we lose the structure
        S = S.reshape(*S.shape[:-2], S.shape[-2:].numel())
        # can assume that block_dim is -3 here
        U = self.__class__(U)
        V = self.__class__(V)
        return U, S, V

    def _symeig(
        self: LinearOperator,  # shape: (*batch, N, N)
        eigenvectors: bool = False,
        return_evals_as_lazy: Optional[bool] = False,
    ) -> Tuple[Tensor, Optional[LinearOperator]]:  # shape: (*batch, M), (*batch, N, M)
        evals, evecs = self.base_linear_op._symeig(eigenvectors=eigenvectors)
        # Doesn't make much sense to sort here, o/w we lose the structure
        evals = evals.reshape(*evals.shape[:-2], evals.shape[-2:].numel())
        if eigenvectors:
            evecs = self.__class__(evecs)  # can assume that block_dim is -3 here
        else:
            evecs = None
        return evals, evecs
