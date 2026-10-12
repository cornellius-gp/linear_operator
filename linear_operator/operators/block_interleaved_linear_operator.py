#!/usr/bin/env python3
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from linear_operator.operators._linear_operator import IndexType, LinearOperator
from linear_operator.operators.block_linear_operator import BlockLinearOperator

from linear_operator.utils.getitem import _is_noop_index, _noop_index
from linear_operator.utils.memoize import cached


class BlockInterleavedLinearOperator(BlockLinearOperator):
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

    @property
    def num_blocks(self) -> int:
        return self.base_linear_op.size(-3)

    def _add_batch_dim(self, other):
        *batch_shape, num_rows, num_cols = other.shape
        batch_shape = list(batch_shape)

        batch_shape.append(num_rows // self.num_blocks)
        other = other.view(*batch_shape, self.num_blocks, num_cols)
        other = other.transpose(-2, -3).contiguous()
        return other

    @cached(name="cholesky")
    def _cholesky(
        self: LinearOperator, upper: bool | None = False  # shape: (*batch, N, N)
    ) -> LinearOperator:  # shape: (*batch, N, N)
        from linear_operator.operators.triangular_linear_operator import TriangularLinearOperator

        chol = self.__class__(self.base_linear_op.cholesky(upper=upper))
        return TriangularLinearOperator(chol, upper=upper)

    def _cholesky_solve(
        self: LinearOperator,  # shape: (*batch, N, N)
        rhs: LinearOperator | Tensor,  # shape: (*batch2, N, M)
        upper: bool | None = False,
    ) -> LinearOperator | Tensor:  # shape: (..., N, M)
        rhs = self._add_batch_dim(rhs)
        res = self.base_linear_op._cholesky_solve(rhs, upper=upper)
        res = self._remove_batch_dim(res)
        return res

    def _diagonal(
        self: LinearOperator,  # shape: (..., M, N)
    ) -> torch.Tensor:  # shape: (..., N)
        block_diag = self.base_linear_op._diagonal()
        return block_diag.mT.contiguous().view(*block_diag.shape[:-2], -1)

    def _get_indices(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> torch.Tensor:
        # Figure out what block the row/column indices belong to
        row_index_block = row_index.fmod(self.base_linear_op.size(-3))
        col_index_block = col_index.fmod(self.base_linear_op.size(-3))

        # Find the row/col index within each block
        row_index = torch.div(row_index, self.base_linear_op.size(-3), rounding_mode="floor")
        col_index = torch.div(col_index, self.base_linear_op.size(-3), rounding_mode="floor")

        # If the row/column blocks do not agree, then we have off diagonal elements
        # These elements should be zeroed out
        res = self.base_linear_op._get_indices(row_index, col_index, *batch_indices, row_index_block)
        res = res * torch.eq(row_index_block, col_index_block).type_as(res)
        return res

    def _getitem(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> LinearOperator:
        # Batch indexing, tensor indices and steps are handled by the base case
        if (_is_noop_index(row_index) and _is_noop_index(col_index)) or not (
            isinstance(row_index, slice) and isinstance(col_index, slice)
        ):
            return super()._getitem(row_index, col_index, *batch_indices)
        num_blocks = self.num_blocks
        num_rows, num_cols = self.matrix_shape
        row_start, row_end, row_step = row_index.start or 0, row_index.stop or num_rows, row_index.step
        col_start, col_end, col_step = col_index.start or 0, col_index.stop or num_cols, col_index.step
        if row_step is not None or col_step is not None:
            return super()._getitem(row_index, col_index, *batch_indices)

        # Row (column) i of the matrix is row (column) i // num_blocks of block i % num_blocks.
        # So if the slice bounds are multiples of num_blocks, the slices select the same rows and
        # columns of every block. Otherwise, use the base case
        if (row_start % num_blocks) or (col_start % num_blocks) or (row_end % num_blocks) or (col_end % num_blocks):
            return super()._getitem(row_index, col_index, *batch_indices)
        row_index = slice(row_start // num_blocks, row_end // num_blocks, None)
        col_index = slice(col_start // num_blocks, col_end // num_blocks, None)
        new_base_linear_op = self.base_linear_op._getitem(row_index, col_index, *batch_indices, _noop_index)
        return self.__class__(new_base_linear_op, block_dim=-3)

    def _remove_batch_dim(self, other):
        other = other.transpose(-2, -3).contiguous()
        shape = list(other.shape)
        del shape[-2]
        shape[-2] *= self.num_blocks
        other = other.reshape(*shape)
        return other

    def _root_decomposition(
        self: LinearOperator,  # shape: (..., N, N)
    ) -> torch.Tensor | LinearOperator:  # shape: (..., N, N)
        return self.__class__(self.base_linear_op._root_decomposition())

    def _root_inv_decomposition(
        self: LinearOperator,  # shape: (*batch, N, N)
        initial_vectors: torch.Tensor | None = None,
        test_vectors: torch.Tensor | None = None,
    ) -> LinearOperator | Tensor:  # shape: (..., N, N)
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
        preconditioner: Callable[[torch.Tensor], torch.Tensor] | None = None,  # shape: (..., N, C)
        num_tridiag: int | None = 0,
    ) -> (
        torch.Tensor  # shape: (..., N, C)
        | tuple[
            torch.Tensor,  # shape: (..., N, C)
            torch.Tensor,  # Note that in case of a tuple the second term size depends on num_tridiag  # shape: (...)
        ]
    ):
        if num_tridiag:
            return super()._solve(rhs, preconditioner, num_tridiag=num_tridiag)
        else:
            rhs = self._add_batch_dim(rhs)
            res = self.base_linear_op._solve(rhs, preconditioner, num_tridiag=None)
            res = self._remove_batch_dim(res)
            return res

    def inv_quad_logdet(
        self: LinearOperator,  # shape: (*batch, N, N)
        inv_quad_rhs: Tensor | None = None,  # shape: (*batch, N, M) or (*batch, N)
        logdet: bool | None = False,
        reduce_inv_quad: bool | None = True,
    ) -> tuple[  # fmt: off
        Tensor | None,  # shape: (*batch, M) or (*batch) or (0)
        Tensor | None,  # shape: (...)
    ]:  # fmt: on
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
