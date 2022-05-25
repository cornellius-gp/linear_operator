#!/usr/bin/env python3

import torch

from ..utils.getitem import _compute_getitem_size
from ..utils.memoize import cached
from ._linear_operator import LinearOperator


class ZeroLinearOperator(LinearOperator):
    """
    Special LinearOperator representing zero.
    """

    def __init__(self, *sizes, dtype=None, device=None):
        super(ZeroLinearOperator, self).__init__(*sizes)
        self.sizes = list(sizes)

        self._dtype = dtype or torch.get_default_dtype()
        self._device = device or torch.device("cpu")

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    def _diagonal(self):
        shape = self.shape
        return torch.zeros(shape[:-1], dtype=self.dtype, device=self.device)

    def _expand_batch(self, batch_shape):
        return self.__class__(*batch_shape, *self.sizes[-2:], dtype=self._dtype, device=self._device)

    def _get_indices(self, row_index, col_index, *batch_indices):
        new_size = _compute_getitem_size(self, batch_indices + (row_index, col_index))
        return ZeroLinearOperator(*new_size)

    def _getitem(self, row_index, col_index, *batch_indices):
        new_size = _compute_getitem_size(self, batch_indices + (row_index, col_index))
        return ZeroLinearOperator(*new_size)

    def _matmul(self, rhs):
        rhs_size_ind = -2 if rhs.ndimension() > 1 else -1
        if self.size(-1) != rhs.size(rhs_size_ind):
            raise RuntimeError("Size mismatch, self: {}, rhs: {}".format(self.size(), rhs.size()))
        new_m = self.size(-2)
        if rhs_size_ind == -1:
            *batch_shape, m = rhs.shape
            output_shape = (*batch_shape, new_m)
        else:
            *batch_shape, m, n = rhs.shape
            output_shape = (*batch_shape, new_m, n)
        return torch.zeros(*output_shape, dtype=rhs.dtype, device=rhs.device)

    def _prod_batch(self, dim):
        sizes = list(self.sizes)
        del sizes[dim]
        return self.__class__(*sizes, dtype=self._dtype, device=self._device)

    def _bilinear_derivative(self, left_vecs, right_vecs):
        raise RuntimeError("Backwards through a ZeroLinearOperator is not possible")

    def _root_decomposition(self):
        raise RuntimeError("ZeroLinearOperators are not positive definite!")

    def _root_inv_decomposition(self, initial_vectors=None):
        raise RuntimeError("ZeroLinearOperators are not positive definite!")

    def _root_decomposition_size(self):
        raise RuntimeError("ZeroLinearOperators are not positive definite!")

    def _size(self):
        return torch.Size(self.sizes)

    def _sum_batch(self, dim):
        sizes = list(self.sizes)
        del sizes[dim]
        return self.__class__(*sizes, dtype=self._dtype, device=self._device)

    def _t_matmul(self, rhs):
        rhs_size_ind = -2 if rhs.ndimension() > 1 else -1
        if self.size(-2) != rhs.size(rhs_size_ind):
            raise RuntimeError("Size mismatch, self: {}, rhs: {}".format(self.size(), rhs.size()))
        new_m = self.size(-1)
        if rhs_size_ind == -1:
            *batch_shape, m = rhs.shape
            output_shape = (*batch_shape, new_m)
        else:
            *batch_shape, m, n = rhs.shape
            output_shape = (*batch_shape, new_m, n)
        return torch.zeros(*output_shape, dtype=rhs.dtype, device=rhs.device)

    def _transpose_nonbatch(self):
        return self.transpose(-2, -1)

    def _unsqueeze_batch(self, dim):
        sizes = self.sizes.copy()
        sizes.insert(dim, 1)
        return self.__class__(*sizes, dtype=self._dtype, device=self._device)

    def add_diagonal(self, diag):
        from .diag_linear_operator import DiagLinearOperator

        if self.size(-1) != self.size(-2):
            raise RuntimeError("add_diag only defined for square matrices")

        if self.ndimension() == 3:
            if diag.ndimension() == 0:
                diag = diag.view(1, 1).expand(self.size(0), self.size(1))
            elif diag.ndimension() == 1:
                diag = diag.unsqueeze(0).expand(self.size(0), self.size(1))
            elif diag.ndimension() == 2:
                diag = diag.expand(self.size(0), self.size(1))
            else:
                raise RuntimeError(
                    "For a 3D tensor ({}), add_diag expects a 1D or 2D diag. "
                    "Got size ({})".format(self.size(), diag.size())
                )
        else:
            if diag.ndimension() == 0:
                diag = diag.view(1).expand(self.size(0))
            elif diag.ndimension() == 1:
                diag = diag.expand(self.size(0))
            else:
                raise RuntimeError(
                    "For a 3D tensor ({}), add_diag expects a 1D or 2D diag. "
                    "Got size ({})".format(self.size(), diag.size())
                )

        res = DiagLinearOperator(diag)
        if res.size() != self.size():
            raise RuntimeError(
                "Diag dimensions are incompatible with the base LinearOperator dimensions. "
                "Diag size corresponds to a {} Tensor - expected {}".format(res.size(), self.size())
            )
        return res

    @cached
    def to_dense(self):
        return torch.zeros(*self.sizes)

    def inv_quad(self, tensor):
        raise RuntimeError("ZeroLinearOperators are not invertible!")

    def inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True):
        raise RuntimeError("ZeroLinearOperators are not invertible!")

    def logdet(self):
        return torch.log(torch.tensor(0.0))

    def matmul(self, tensor):
        tensor_size_ind = -2 if tensor.ndimension() > 1 else -1
        if self.size(-1) != tensor.size(tensor_size_ind):
            raise RuntimeError("Size mismatch, self: {}, tensor: {}".format(self.size(), tensor.size()))
        new_m = self.size(-2)
        if tensor_size_ind == -1:
            *batch_shape, m = tensor.shape
            output_shape = (*batch_shape, new_m)
        else:
            *batch_shape, m, n = tensor.shape
            output_shape = (*batch_shape, new_m, n)
        return ZeroLinearOperator(*output_shape, dtype=tensor.dtype, device=tensor.device)

    def mul(self, other):
        shape = torch.broadcast_shapes(self.shape, other.shape)
        return self.__class__(*shape, dtype=self._dtype, device=self._device)

    def solve(self, right_tensor, left_tensor=None):
        raise RuntimeError("ZeroLinearOperators are not invertible!")

    def transpose(self, dim1, dim2):
        sizes = self.sizes.copy()
        tmp = sizes[dim1]
        sizes[dim1] = sizes[dim2]
        sizes[dim2] = tmp

        return ZeroLinearOperator(*sizes)

    def __add__(self, other):
        return other

    def __div__(self, other):
        return self

    def __mul__(self, other):
        return self
