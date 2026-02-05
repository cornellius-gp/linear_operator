#!/usr/bin/env python3
from __future__ import annotations

import torch
from torch import Tensor

from linear_operator.operators._linear_operator import IndexType, LinearOperator
from linear_operator.operators.dense_linear_operator import DenseLinearOperator, to_linear_operator
from linear_operator.operators.matmul_linear_operator import MatmulLinearOperator

from linear_operator.utils.broadcasting import _pad_with_singletons
from linear_operator.utils.getitem import _equal_indices, _noop_index
from linear_operator.utils.memoize import cached


class RootLinearOperator(LinearOperator):
    def __init__(self, root):
        root = to_linear_operator(root)
        super().__init__(root)
        self.root = root

    def _diagonal(
        self: LinearOperator,  # shape: (..., M, N)
    ) -> torch.Tensor:  # shape: (..., N)
        if isinstance(self.root, DenseLinearOperator):
            return (self.root.tensor**2).sum(-1)
        else:
            return super()._diagonal()

    def _expand_batch(
        self: LinearOperator, batch_shape: torch.Size | list[int]  # shape: (..., M, N)
    ) -> LinearOperator:  # shape: (..., M, N)
        if len(batch_shape) == 0:
            return self
        return self.__class__(self.root._expand_batch(batch_shape))

    def _get_indices(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> torch.Tensor:
        row_index = row_index.unsqueeze(-1)
        col_index = col_index.unsqueeze(-1)
        batch_indices = tuple(batch_index.unsqueeze(-1) for batch_index in batch_indices)
        inner_index = torch.arange(0, self.root.size(-1), device=self.device)
        inner_index = _pad_with_singletons(inner_index, row_index.dim() - 1, 0)

        left_tensor = self.root._get_indices(row_index, inner_index, *batch_indices)
        if torch.equal(row_index, col_index):
            res = left_tensor.pow(2).sum(-1)
        else:
            right_tensor = self.root._get_indices(col_index, inner_index, *batch_indices)
            res = (left_tensor * right_tensor).sum(-1)
        return res

    def _getitem(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> LinearOperator:
        # Make sure we're not generating more memory with our "efficient" method
        if torch.is_tensor(row_index) and torch.is_tensor(col_index):
            num_indices = row_index.numel()
            if num_indices > self.matrix_shape.numel():
                return to_linear_operator(self.to_dense())._getitem(row_index, col_index, *batch_indices)

        left_tensor = self.root._getitem(row_index, _noop_index, *batch_indices)
        if _equal_indices(row_index, col_index):
            res = self.__class__(left_tensor)
        else:
            right_tensor = self.root._getitem(col_index, _noop_index, *batch_indices)
            res = MatmulLinearOperator(left_tensor, right_tensor.mT)

        return res

    def _matmul(
        self: LinearOperator,  # shape: (*batch, M, N)
        rhs: torch.Tensor,  # shape: (*batch2, N, C) or (*batch2, N)
    ) -> torch.Tensor:  # shape: (..., M, C) or (..., M)
        return self.root._matmul(self.root._t_matmul(rhs))

    def _mul_constant(
        self: LinearOperator, other: float | torch.Tensor  # shape: (*batch, M, N)
    ) -> LinearOperator:  # shape: (*batch, M, N)
        if (other > 0).all():
            res = self.__class__(self.root._mul_constant(other.sqrt()))
        else:
            res = super()._mul_constant(other)
        return res

    def _t_matmul(
        self: LinearOperator,  # shape: (*batch, M, N)
        rhs: Tensor | LinearOperator,  # shape: (*batch2, M, P)
    ) -> LinearOperator | Tensor:  # shape: (..., N, P)
        # Matrix is symmetric
        return self._matmul(rhs)

    def add_low_rank(
        self: LinearOperator,  # shape: (*batch, N, N)
        low_rank_mat: Tensor | LinearOperator,  # shape: (..., N, _)
        root_decomp_method: str | None = None,
        root_inv_decomp_method: str | None = None,
        generate_roots: bool | None = True,
        **root_decomp_kwargs,
    ) -> LinearOperator:  # shape: (*batch, N, N)
        return super().add_low_rank(low_rank_mat, root_inv_decomp_method=root_inv_decomp_method)

    def root_decomposition(
        self: LinearOperator, method: str | None = None  # shape: (*batch, N, N)
    ) -> LinearOperator:  # shape: (*batch, N, N)
        return self

    def _root_decomposition(
        self: LinearOperator,  # shape: (..., N, N)
    ) -> torch.Tensor | LinearOperator:  # shape: (..., N, N)
        return self.root

    def _root_decomposition_size(self) -> int:
        return self.root.size(-1)

    def _size(self) -> torch.Size:
        return torch.Size((*self.root.batch_shape, self.root.size(-2), self.root.size(-2)))

    def _transpose_nonbatch(
        self: LinearOperator,  # shape: (*batch, M, N)
    ) -> LinearOperator:  # shape: (*batch, N, M)
        return self

    @cached
    def to_dense(
        self: LinearOperator,  # shape: (*batch, M, N)
    ) -> Tensor:  # shape: (*batch, M, N)
        eval_root = self.root.to_dense()
        return torch.matmul(eval_root, eval_root.mT)
