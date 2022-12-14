#!/usr/bin/env python3
from typing import Optional, Union

import torch
from jaxtyping import Float
from torch import Tensor

from ..utils.broadcasting import _pad_with_singletons
from ..utils.getitem import _equal_indices, _noop_index
from ..utils.memoize import cached
from ._linear_operator import IndexType, LinearOperator
from .dense_linear_operator import DenseLinearOperator, to_linear_operator
from .matmul_linear_operator import MatmulLinearOperator


class RootLinearOperator(LinearOperator):
    def __init__(self, root):
        root = to_linear_operator(root)
        super().__init__(root)
        self.root = root

    def _diagonal(self: Float[LinearOperator, "*batch N N"]) -> Float[torch.Tensor, "... N"]:
        if isinstance(self.root, DenseLinearOperator):
            return (self.root.tensor**2).sum(-1)
        else:
            return super()._diagonal()

    def _expand_batch(
        self: Float[LinearOperator, "... M N"], batch_shape: torch.Size
    ) -> Float[LinearOperator, "... M N"]:
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

    def _getitem(
        self,
        row_index: IndexType,
        col_index: IndexType,
        *batch_indices: IndexType,
    ) -> LinearOperator:
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
        self: Float[LinearOperator, "*batch M N"],
        rhs: Union[Float[torch.Tensor, "*batch2 N C"], Float[torch.Tensor, "*batch2 N"]],
    ) -> Union[Float[torch.Tensor, "... M C"], Float[torch.Tensor, "... M"]]:
        return self.root._matmul(self.root._t_matmul(rhs))

    def _mul_constant(
        self: Float[LinearOperator, "*batch M N"], constant: Union[float, torch.Tensor]
    ) -> Float[LinearOperator, "*batch M N"]:
        if (constant > 0).all():
            res = self.__class__(self.root._mul_constant(constant.sqrt()))
        else:
            res = super()._mul_constant(constant)
        return res

    def _t_matmul(
        self: Float[LinearOperator, "*batch M N"],
        rhs: Union[Float[Tensor, "*batch2 M P"], Float[LinearOperator, "*batch2 M P"]],
    ) -> Union[Float[LinearOperator, "... N P"], Float[Tensor, "... N P"]]:
        # Matrix is symmetric
        return self._matmul(rhs)

    def add_low_rank(
        self: Float[LinearOperator, "*batch N N"],
        low_rank_mat: Union[Float[Tensor, "... N _"], Float[LinearOperator, "... N _"]],
        root_decomp_method: Optional[str] = None,
        root_inv_decomp_method: Optional[str] = None,
        generate_roots: Optional[bool] = True,
        **root_decomp_kwargs,
    ) -> Float[LinearOperator, "*batch N N"]:
        return super().add_low_rank(low_rank_mat, root_inv_decomp_method=root_inv_decomp_method)

    def root_decomposition(self, method=None):
        return self

    def _root_decomposition(self):
        return self.root

    def _root_decomposition_size(self):
        return self.root.size(-1)

    def _size(self) -> torch.Size:
        return torch.Size((*self.root.batch_shape, self.root.size(-2), self.root.size(-2)))

    def _transpose_nonbatch(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch N M"]:
        return self

    @cached
    def to_dense(self):
        eval_root = self.root.to_dense()
        return torch.matmul(eval_root, eval_root.mT)
