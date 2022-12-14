#!/usr/bin/env python3

from typing import Optional, Tuple, Union

import torch
from jaxtyping import Float
from torch import Tensor

from ..utils.broadcasting import _matmul_broadcast_shape, _pad_with_singletons
from ..utils.getitem import _noop_index
from ..utils.memoize import cached
from ._linear_operator import IndexType, LinearOperator
from .dense_linear_operator import DenseLinearOperator, to_linear_operator
from .diag_linear_operator import DiagLinearOperator


def _inner_repeat(tensor, amt):
    return tensor.unsqueeze(-1).repeat(amt, 1).squeeze(-1)


def _outer_repeat(tensor, amt):
    return tensor.unsqueeze(-1).repeat(1, amt).view(-1)


class MatmulLinearOperator(LinearOperator):
    def __init__(self, left_linear_op, right_linear_op):
        left_linear_op = to_linear_operator(left_linear_op)
        right_linear_op = to_linear_operator(right_linear_op)

        # Match batch dimensions
        batch_shape = torch.broadcast_shapes(left_linear_op.batch_shape, right_linear_op.batch_shape)
        if left_linear_op.batch_shape != batch_shape:
            left_linear_op = left_linear_op._expand_batch(batch_shape)
        if right_linear_op.batch_shape != batch_shape:
            right_linear_op = right_linear_op._expand_batch(batch_shape)

        super().__init__(left_linear_op, right_linear_op)
        batch_shape = torch.broadcast_shapes(left_linear_op.batch_shape, right_linear_op.batch_shape)
        if left_linear_op.batch_shape != batch_shape:
            self.left_linear_op = left_linear_op._expand_batch(batch_shape)
        else:
            self.left_linear_op = left_linear_op
        if right_linear_op.batch_shape != batch_shape:
            self.right_linear_op = right_linear_op._expand_batch(batch_shape)
        else:
            self.right_linear_op = right_linear_op

    def _expand_batch(
        self: Float[LinearOperator, "... M N"], batch_shape: torch.Size
    ) -> Float[LinearOperator, "... M N"]:
        return self.__class__(
            self.left_linear_op._expand_batch(batch_shape), self.right_linear_op._expand_batch(batch_shape)
        )

    def _get_indices(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> torch.Tensor:
        row_index = row_index.unsqueeze(-1)
        col_index = col_index.unsqueeze(-1)
        batch_indices = tuple(batch_index.unsqueeze(-1) for batch_index in batch_indices)
        inner_index = torch.arange(0, self.left_linear_op.size(-1), device=self.device)
        inner_index = _pad_with_singletons(inner_index, row_index.dim() - 1, 0)

        left_tensor = self.left_linear_op._get_indices(
            row_index, inner_index, *batch_indices[-len(self.left_linear_op.batch_shape) :]
        )
        right_tensor = self.right_linear_op._get_indices(
            inner_index, col_index, *batch_indices[-len(self.right_linear_op.batch_shape) :]
        )
        res = (left_tensor * right_tensor).sum(-1)
        return res

    def _diagonal(self: Float[LinearOperator, "*batch N N"]) -> Float[torch.Tensor, "... N"]:
        if isinstance(self.left_linear_op, DenseLinearOperator) and isinstance(
            self.right_linear_op, DenseLinearOperator
        ):
            return (self.left_linear_op.tensor * self.right_linear_op.tensor.mT).sum(-1)
        elif isinstance(self.left_linear_op, DiagLinearOperator) or isinstance(
            self.right_linear_op, DiagLinearOperator
        ):
            return self.left_linear_op._diagonal() * self.right_linear_op._diagonal()
        else:
            return super()._diagonal()

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

        left_tensor = self.left_linear_op._getitem(row_index, _noop_index, *batch_indices)
        right_tensor = self.right_linear_op._getitem(_noop_index, col_index, *batch_indices)

        res = MatmulLinearOperator(left_tensor, right_tensor)
        return res

    def _matmul(
        self: Float[LinearOperator, "*batch M N"],
        rhs: Union[Float[torch.Tensor, "*batch2 N C"], Float[torch.Tensor, "*batch2 N"]],
    ) -> Union[Float[torch.Tensor, "... M C"], Float[torch.Tensor, "... M"]]:
        return self.left_linear_op._matmul(self.right_linear_op._matmul(rhs))

    def _t_matmul(
        self: Float[LinearOperator, "*batch M N"],
        rhs: Union[Float[Tensor, "*batch2 M P"], Float[LinearOperator, "*batch2 M P"]],
    ) -> Union[Float[LinearOperator, "... N P"], Float[Tensor, "... N P"]]:
        return self.right_linear_op._t_matmul(self.left_linear_op._t_matmul(rhs))

    def _bilinear_derivative(self, left_vecs: Tensor, right_vecs: Tensor) -> Tuple[Optional[Tensor], ...]:
        if left_vecs.ndimension() == 1:
            left_vecs = left_vecs.unsqueeze(1)
            right_vecs = right_vecs.unsqueeze(1)
        right_vecs_times_right_linear_op = self.right_linear_op._matmul(right_vecs)
        left_vecs_times_left_linear_op_t = self.left_linear_op._t_matmul(left_vecs)
        left_grad = self.left_linear_op._bilinear_derivative(left_vecs, right_vecs_times_right_linear_op)
        right_grad = self.right_linear_op._bilinear_derivative(left_vecs_times_left_linear_op_t, right_vecs)

        left_grad = (left_grad,) if not isinstance(left_grad, tuple) else left_grad
        right_grad = (right_grad,) if not isinstance(right_grad, tuple) else right_grad
        return left_grad + right_grad

    def _permute_batch(self, *dims: int) -> LinearOperator:
        return self.__class__(self.left_linear_op._permute_batch(*dims), self.right_linear_op._permute_batch(*dims))

    def _size(self) -> torch.Size:
        return _matmul_broadcast_shape(self.left_linear_op.shape, self.right_linear_op.shape)

    def _transpose_nonbatch(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch N M"]:
        return self.__class__(self.right_linear_op._transpose_nonbatch(), self.left_linear_op._transpose_nonbatch())

    @cached
    def to_dense(self):
        return torch.matmul(self.left_linear_op.to_dense(), self.right_linear_op.to_dense())
