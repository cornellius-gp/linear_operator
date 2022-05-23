#!/usr/bin/env python3
from torch import Tensor

from ..utils.broadcasting import _mul_broadcast_shape
from ..utils.memoize import cached
from ._linear_operator import LinearOperator
from .dense_linear_operator import lazify
from .zero_linear_operator import ZeroLinearOperator

# from .broadcasted_linear_operator import BroadcastedLinearOperator


class SumLinearOperator(LinearOperator):
    def __init__(self, *lazy_tensors, **kwargs):
        try:
            lazy_tensors = tuple(lazify(lt) for lt in lazy_tensors)
        except TypeError:
            raise TypeError("All arguments of a SumLinearOperator should be LinearOperators or Tensors")
        batch_shape = _mul_broadcast_shape(*[lt.batch_shape for lt in lazy_tensors])
        lazy_tensors = tuple(
            lt._expand_batch(batch_shape) if lt.batch_shape != batch_shape else lt for lt in lazy_tensors
        )
        super(SumLinearOperator, self).__init__(*lazy_tensors, **kwargs)

        self.lazy_tensors = lazy_tensors

    def _expand_batch(self, batch_shape):
        expanded_tensors = [lazy_tensor._expand_batch(batch_shape) for lazy_tensor in self.lazy_tensors]
        return self.__class__(*expanded_tensors)

    def _get_indices(self, row_index, col_index, *batch_indices):
        results = [lazy_tensor._get_indices(row_index, col_index, *batch_indices) for lazy_tensor in self.lazy_tensors]
        return sum(results)

    def _getitem(self, row_index, col_index, *batch_indices):
        results = [lazy_tensor._getitem(row_index, col_index, *batch_indices) for lazy_tensor in self.lazy_tensors]
        return SumLinearOperator(*results)

    def _matmul(self, rhs):
        return sum(lazy_tensor._matmul(rhs) for lazy_tensor in self.lazy_tensors)

    def _mul_constant(self, other):
        # We're using a custom method here - the constant mul is applied to the base_lazy_tensors
        return self.__class__(*[lt._mul_constant(other) for lt in self.lazy_tensors])

    def _quad_form_derivative(self, left_vecs, right_vecs):
        return tuple(
            var for lazy_tensor in self.lazy_tensors for var in lazy_tensor._quad_form_derivative(left_vecs, right_vecs)
        )

    def _size(self):
        return _mul_broadcast_shape(*[lt.shape for lt in self.lazy_tensors])

    def _sum_batch(self, dim):
        return self.__class__(*(lazy_tensor._sum_batch(dim) for lazy_tensor in self.lazy_tensors))

    def _t_matmul(self, rhs):
        return sum(lazy_tensor._t_matmul(rhs) for lazy_tensor in self.lazy_tensors)

    def _transpose_nonbatch(self):
        lazy_tensors_t = [lazy_tensor.transpose(-1, -2) for lazy_tensor in self.lazy_tensors]
        return self.__class__(*lazy_tensors_t)

    @cached
    def evaluate(self):
        return sum(lazy_tensor.evaluate() for lazy_tensor in self.lazy_tensors)

    def __add__(self, other):
        from .added_diag_linear_operator import AddedDiagLinearOperator
        from .diag_linear_operator import DiagLinearOperator

        if isinstance(other, ZeroLinearOperator):
            return self
        elif isinstance(other, DiagLinearOperator):
            return AddedDiagLinearOperator(self, other)
        elif isinstance(other, SumLinearOperator):
            return SumLinearOperator(*(list(self.lazy_tensors) + list(other.lazy_tensors)))
        elif isinstance(other, LinearOperator):
            return SumLinearOperator(*(list(self.lazy_tensors) + [other]))
        elif isinstance(other, Tensor):
            # get broadcast shape, assuming mul broadcasting the same as add broadcasting
            broadcasted_shape = _mul_broadcast_shape(self.shape, other.shape)

            # lazify + broadcast other
            broadcasted_other = lazify(other.expand(broadcasted_shape))

            # update the lazy tensors' shape as well
            new_self = self if broadcasted_shape == self.shape else self._expand_batch(broadcasted_shape[:-2])

            return SumLinearOperator(*(list(new_self.lazy_tensors) + [broadcasted_other]))
        else:
            raise AttributeError("other must be a LinearOperator")

    def diag(self):
        return sum(lazy_tensor.diag().contiguous() for lazy_tensor in self.lazy_tensors)
