#!/usr/bin/env python3
from __future__ import annotations

import torch
from torch import Tensor

from linear_operator.operators._linear_operator import IndexType, LinearOperator
from linear_operator.operators.dense_linear_operator import to_linear_operator
from linear_operator.operators.zero_linear_operator import ZeroLinearOperator

from linear_operator.utils.memoize import cached

# from linear_operator.operators.broadcasted_linear_operator import BroadcastedLinearOperator


class SumLinearOperator(LinearOperator):
    def __init__(self, *linear_ops, **kwargs):
        try:
            linear_ops = tuple(to_linear_operator(lt) for lt in linear_ops)
        except TypeError:
            raise TypeError("All arguments of a SumLinearOperator should be LinearOperators or Tensors")
        batch_shape = torch.broadcast_shapes(*[lt.batch_shape for lt in linear_ops])
        linear_ops = tuple(lt._expand_batch(batch_shape) if lt.batch_shape != batch_shape else lt for lt in linear_ops)
        super(SumLinearOperator, self).__init__(*linear_ops, **kwargs)

        self.linear_ops = linear_ops

    def _diagonal(
        self: LinearOperator,  # shape: (..., M, N)
    ) -> torch.Tensor:  # shape: (..., N)
        return sum(linear_op._diagonal().contiguous() for linear_op in self.linear_ops)

    def _expand_batch(
        self: LinearOperator, batch_shape: torch.Size | list[int]  # shape: (..., M, N)
    ) -> LinearOperator:  # shape: (..., M, N)
        expanded_tensors = [linear_op._expand_batch(batch_shape) for linear_op in self.linear_ops]
        return self.__class__(*expanded_tensors)

    def _get_indices(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> torch.Tensor:
        results = [linear_op._get_indices(row_index, col_index, *batch_indices) for linear_op in self.linear_ops]
        return sum(results)

    def _getitem(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> LinearOperator:
        results = [linear_op._getitem(row_index, col_index, *batch_indices) for linear_op in self.linear_ops]
        return SumLinearOperator(*results)

    def _matmul(
        self: LinearOperator,  # shape: (*batch, M, N)
        rhs: torch.Tensor,  # shape: (*batch2, N, C) or (*batch2, N)
    ) -> torch.Tensor:  # shape: (..., M, C) or (..., M)
        return sum(linear_op._matmul(rhs) for linear_op in self.linear_ops)

    def _mul_constant(
        self: LinearOperator, other: float | torch.Tensor  # shape: (*batch, M, N)
    ) -> LinearOperator:  # shape: (*batch, M, N)
        # We're using a custom method here - the constant mul is applied to the base_linear_ops
        return self.__class__(*[lt._mul_constant(other) for lt in self.linear_ops])

    def _bilinear_derivative(self, left_vecs: Tensor, right_vecs: Tensor) -> tuple[Tensor | None, ...]:
        return tuple(
            var for linear_op in self.linear_ops for var in linear_op._bilinear_derivative(left_vecs, right_vecs)
        )

    def _size(self) -> torch.Size:
        return torch.broadcast_shapes(*[lt.shape for lt in self.linear_ops])

    def _sum_batch(self, dim: int) -> LinearOperator:
        return self.__class__(*(linear_op._sum_batch(dim) for linear_op in self.linear_ops))

    def _t_matmul(
        self: LinearOperator,  # shape: (*batch, M, N)
        rhs: Tensor | LinearOperator,  # shape: (*batch2, M, P)
    ) -> LinearOperator | Tensor:  # shape: (..., N, P)
        return sum(linear_op._t_matmul(rhs) for linear_op in self.linear_ops)

    def _transpose_nonbatch(
        self: LinearOperator,  # shape: (*batch, M, N)
    ) -> LinearOperator:  # shape: (*batch, N, M)
        linear_ops_t = [linear_op.mT for linear_op in self.linear_ops]
        return self.__class__(*linear_ops_t)

    @cached
    def to_dense(
        self: LinearOperator,  # shape: (*batch, M, N)
    ) -> Tensor:  # shape: (*batch, M, N)
        return (sum(linear_op.to_dense() for linear_op in self.linear_ops)).contiguous()

    def __add__(
        self: LinearOperator,  # shape: (..., #M, #N)
        other: Tensor | LinearOperator | float,  # shape: (..., #M, #N)
    ) -> LinearOperator | Tensor:  # shape: (..., M, N)
        from linear_operator.operators.added_diag_linear_operator import AddedDiagLinearOperator
        from linear_operator.operators.diag_linear_operator import DiagLinearOperator

        match other:
            case ZeroLinearOperator():
                return self
            case DiagLinearOperator():
                return AddedDiagLinearOperator(self, other)
            case SumLinearOperator():
                return SumLinearOperator(*(list(self.linear_ops) + list(other.linear_ops)))
            case LinearOperator():
                return SumLinearOperator(*(list(self.linear_ops) + [other]))
            case Tensor():
                # get broadcast shape, assuming mul broadcasting the same as add broadcasting
                broadcasted_shape = torch.broadcast_shapes(self.shape, other.shape)

                # to_linear_operator + broadcast other
                broadcasted_other = to_linear_operator(other.expand(broadcasted_shape))

                # update the lazy tensors' shape as well
                new_self = self if broadcasted_shape == self.shape else self._expand_batch(broadcasted_shape[:-2])

                return SumLinearOperator(*(list(new_self.linear_ops) + [broadcasted_other]))
            case _:
                raise AttributeError("other must be a LinearOperator")
