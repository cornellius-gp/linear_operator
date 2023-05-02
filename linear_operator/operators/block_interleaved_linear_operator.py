#!/usr/bin/env python3
from typing import Callable, Optional, Tuple, Union

import torch
from jaxtyping import Float
from torch import Tensor

from ..utils.memoize import cached
from ._linear_operator import IndexType, LinearOperator
from .block_linear_operator import BlockLinearOperator


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
        self: Float[LinearOperator, "*batch N N"], upper: Optional[bool] = False
    ) -> Float[LinearOperator, "*batch N N"]:
        from .triangular_linear_operator import TriangularLinearOperator

        chol = self.__class__(self.base_linear_op.cholesky(upper=upper))
        return TriangularLinearOperator(chol, upper=upper)

    def _cholesky_solve(
        self: Float[LinearOperator, "*batch N N"],
        rhs: Union[Float[LinearOperator, "*batch2 N M"], Float[Tensor, "*batch2 N M"]],
        upper: Optional[bool] = False,
    ) -> Union[Float[LinearOperator, "... N M"], Float[Tensor, "... N M"]]:
        rhs = self._add_batch_dim(rhs)
        res = self.base_linear_op._cholesky_solve(rhs, upper=upper)
        res = self._remove_batch_dim(res)
        return res

    def _diagonal(self: Float[LinearOperator, "... M N"]) -> Float[torch.Tensor, "... N"]:
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

    def _remove_batch_dim(self, other):
        other = other.transpose(-2, -3).contiguous()
        shape = list(other.shape)
        del shape[-2]
        shape[-2] *= self.num_blocks
        other = other.reshape(*shape)
        return other

    def _root_decomposition(
        self: Float[LinearOperator, "... N N"]
    ) -> Union[Float[torch.Tensor, "... N N"], Float[LinearOperator, "... N N"]]:
        return self.__class__(self.base_linear_op._root_decomposition())

    def _root_inv_decomposition(
        self: Float[LinearOperator, "*batch N N"],
        initial_vectors: Optional[torch.Tensor] = None,
        test_vectors: Optional[torch.Tensor] = None,
    ) -> Union[Float[LinearOperator, "... N N"], Float[Tensor, "... N N"]]:
        return self.__class__(self.base_linear_op._root_inv_decomposition(initial_vectors))

    def _size(self) -> torch.Size:
        shape = list(self.base_linear_op.shape)
        shape[-2] *= shape[-3]
        shape[-1] *= shape[-3]
        del shape[-3]
        return torch.Size(shape)

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
        else:
            rhs = self._add_batch_dim(rhs)
            res = self.base_linear_op._solve(rhs, preconditioner, num_tridiag=None)
            res = self._remove_batch_dim(res)
            return res

    def inv_quad_logdet(
        self: Float[LinearOperator, "*batch N N"],
        inv_quad_rhs: Optional[Union[Float[Tensor, "*batch N M"], Float[Tensor, "*batch N"]]] = None,
        logdet: Optional[bool] = False,
        reduce_inv_quad: Optional[bool] = True,
    ) -> Tuple[
        Optional[Union[Float[Tensor, "*batch M"], Float[Tensor, " *batch"], Float[Tensor, " 0"]]],
        Optional[Float[Tensor, "..."]],
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
