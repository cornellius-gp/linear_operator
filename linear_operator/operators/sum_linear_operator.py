#!/usr/bin/env python3
from typing import List, Optional, Tuple, Union

import torch
from jaxtyping import Float
from torch import Tensor

from ..utils.memoize import cached
from ._linear_operator import IndexType, LinearOperator
from .dense_linear_operator import to_linear_operator
from .zero_linear_operator import ZeroLinearOperator

# from .broadcasted_linear_operator import BroadcastedLinearOperator


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

    def _diagonal(self: Float[LinearOperator, "... M N"]) -> Float[torch.Tensor, "... N"]:
        return sum(linear_op._diagonal().contiguous() for linear_op in self.linear_ops)

    def _expand_batch(
        self: Float[LinearOperator, "... M N"], batch_shape: Union[torch.Size, List[int]]
    ) -> Float[LinearOperator, "... M N"]:
        expanded_tensors = [linear_op._expand_batch(batch_shape) for linear_op in self.linear_ops]
        return self.__class__(*expanded_tensors)

    def _get_indices(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> torch.Tensor:
        results = [linear_op._get_indices(row_index, col_index, *batch_indices) for linear_op in self.linear_ops]
        return sum(results)

    def _getitem(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> LinearOperator:
        results = [linear_op._getitem(row_index, col_index, *batch_indices) for linear_op in self.linear_ops]
        return SumLinearOperator(*results)

    def _matmul(
        self: Float[LinearOperator, "*batch M N"],
        rhs: Union[Float[torch.Tensor, "*batch2 N C"], Float[torch.Tensor, "*batch2 N"]],
    ) -> Union[Float[torch.Tensor, "... M C"], Float[torch.Tensor, "... M"]]:
        return sum(linear_op._matmul(rhs) for linear_op in self.linear_ops)

    def _mul_constant(
        self: Float[LinearOperator, "*batch M N"], other: Union[float, torch.Tensor]
    ) -> Float[LinearOperator, "*batch M N"]:
        # We're using a custom method here - the constant mul is applied to the base_linear_ops
        return self.__class__(*[lt._mul_constant(other) for lt in self.linear_ops])

    def _bilinear_derivative(self, left_vecs: Tensor, right_vecs: Tensor) -> Tuple[Optional[Tensor], ...]:
        return tuple(
            var for linear_op in self.linear_ops for var in linear_op._bilinear_derivative(left_vecs, right_vecs)
        )

    def _size(self) -> torch.Size:
        return torch.broadcast_shapes(*[lt.shape for lt in self.linear_ops])

    def _sum_batch(self, dim: int) -> LinearOperator:
        return self.__class__(*(linear_op._sum_batch(dim) for linear_op in self.linear_ops))

    def _t_matmul(
        self: Float[LinearOperator, "*batch M N"],
        rhs: Union[Float[Tensor, "*batch2 M P"], Float[LinearOperator, "*batch2 M P"]],
    ) -> Union[Float[LinearOperator, "... N P"], Float[Tensor, "... N P"]]:
        return sum(linear_op._t_matmul(rhs) for linear_op in self.linear_ops)

    def _transpose_nonbatch(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch N M"]:
        linear_ops_t = [linear_op.mT for linear_op in self.linear_ops]
        return self.__class__(*linear_ops_t)

    @cached
    def to_dense(self: Float[LinearOperator, "*batch M N"]) -> Float[Tensor, "*batch M N"]:
        return (sum(linear_op.to_dense() for linear_op in self.linear_ops)).contiguous()

    def __add__(
        self: Float[LinearOperator, "... #M #N"],
        other: Union[Float[Tensor, "... #M #N"], Float[LinearOperator, "... #M #N"], float],
    ) -> Union[Float[LinearOperator, "... M N"], Float[Tensor, "... M N"]]:
        from .added_diag_linear_operator import AddedDiagLinearOperator
        from .diag_linear_operator import DiagLinearOperator

        if isinstance(other, ZeroLinearOperator):
            return self
        elif isinstance(other, DiagLinearOperator):
            return AddedDiagLinearOperator(self, other)
        elif isinstance(other, SumLinearOperator):
            return SumLinearOperator(*(list(self.linear_ops) + list(other.linear_ops)))
        elif isinstance(other, LinearOperator):
            return SumLinearOperator(*(list(self.linear_ops) + [other]))
        elif isinstance(other, Tensor):
            # get broadcast shape, assuming mul broadcasting the same as add broadcasting
            broadcasted_shape = torch.broadcast_shapes(self.shape, other.shape)

            # to_linear_operator + broadcast other
            broadcasted_other = to_linear_operator(other.expand(broadcasted_shape))

            # update the lazy tensors' shape as well
            new_self = self if broadcasted_shape == self.shape else self._expand_batch(broadcasted_shape[:-2])

            return SumLinearOperator(*(list(new_self.linear_ops) + [broadcasted_other]))
        else:
            raise AttributeError("other must be a LinearOperator")
