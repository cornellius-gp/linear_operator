#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from ..utils.getitem import _compute_getitem_size
from ..utils.memoize import cached
from ._linear_operator import LinearOperator


class ZeroLinearOperator(LinearOperator):
    """
    Special LinearOperator representing zero.

    :param sizes: The size of each dimension (including batch dimensions).
    :param dtype: Dtype that the LinearOperator will be operating on. (Default: :meth:`torch.get_default_dtype()`).
    :param device: Device that the LinearOperator will be operating on. (Default: CPU).
    """

    def __init__(
        self, *sizes: Tuple[int, ...], dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None,
        **kwargs
    ):
        super(ZeroLinearOperator, self).__init__(*sizes, dtype=dtype, device=device, **kwargs)
        self.sizes = list(sizes)

        self._dtype = dtype or torch.get_default_dtype()
        self._device = device or torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    def _bilinear_derivative(self, left_vecs: Tensor, right_vecs: Tensor) -> Tuple[Tensor, ...]:
        raise RuntimeError("Backwards through a ZeroLinearOperator is not possible")

    def _diagonal(self) -> Tensor:
        shape = self.shape
        return torch.zeros(shape[:-1], dtype=self.dtype, device=self.device)

    def _expand_batch(self, batch_shape: torch.Size) -> LinearOperator:
        return self.__class__(*batch_shape, *self.sizes[-2:], dtype=self._dtype, device=self._device)

    def _get_indices(
        self, row_index: torch.LongTensor, col_index: torch.LongTensor, *batch_indices: Tuple[torch.LongTensor, ...]
    ) -> Tensor:
        new_size = _compute_getitem_size(self, batch_indices + (row_index, col_index))
        return ZeroLinearOperator(*new_size)

    def _getitem(
        self,
        row_index: Union[slice, torch.LongTensor],
        col_index: Union[slice, torch.LongTensor],
        *batch_indices: Tuple[Union[int, slice, torch.LongTensor], ...],
    ) -> LinearOperator:
        new_size = _compute_getitem_size(self, batch_indices + (row_index, col_index))
        return ZeroLinearOperator(*new_size)

    def _matmul(self, rhs: Tensor) -> Tensor:
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

    def _prod_batch(self, dim: int) -> LinearOperator:
        sizes = list(self.sizes)
        del sizes[dim]
        return self.__class__(*sizes, dtype=self._dtype, device=self._device)

    def _root_decomposition(self) -> Union[Tensor, "LinearOperator"]:
        raise RuntimeError("ZeroLinearOperators are not positive definite!")

    def _root_decomposition_size(self) -> int:
        raise RuntimeError("ZeroLinearOperators are not positive definite!")

    def _root_inv_decomposition(
        self,
        initial_vectors: Optional[Tensor] = None,
        test_vectors: Optional[Tensor] = None,
    ) -> LinearOperator:
        raise RuntimeError("ZeroLinearOperators are not positive definite!")

    def _size(self) -> torch.Size:
        return torch.Size(self.sizes)

    def _sum_batch(self, dim: int) -> LinearOperator:
        sizes = list(self.sizes)
        del sizes[dim]
        return self.__class__(*sizes, dtype=self._dtype, device=self._device)

    def _t_matmul(self, rhs: Tensor) -> LinearOperator:
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

    def _transpose_nonbatch(self) -> LinearOperator:
        return self.mT

    def _unsqueeze_batch(self, dim: int) -> LinearOperator:
        sizes = self.sizes.copy()
        sizes.insert(dim, 1)
        return self.__class__(*sizes, dtype=self._dtype, device=self._device)

    def add_diagonal(self, diag: Tensor) -> LinearOperator:
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

    def div(self, other: Union[float, Tensor]) -> LinearOperator:
        return self

    def inv_quad(self, inv_quad_rhs: Tensor, reduce_inv_quad: bool = True) -> Tensor:
        raise RuntimeError("ZeroLinearOperators are not invertible!")

    def inv_quad_logdet(
        self, inv_quad_rhs: Optional[Tensor] = None, logdet: bool = False, reduce_inv_quad: bool = True
    ) -> Tuple[Tensor, Tensor]:
        raise RuntimeError("ZeroLinearOperators are not invertible!")

    def logdet(self) -> Tensor:
        return torch.log(torch.zeros(sizes[:-2], device=self._device, dtype=self._dtype))

    def matmul(self, other: Union[Tensor, LinearOperator]) -> Union[Tensor, LinearOperator]:
        tensor_size_ind = -2 if other.ndimension() > 1 else -1
        if self.size(-1) != other.size(tensor_size_ind):
            raise RuntimeError("Size mismatch, self: {}, other: {}".format(self.size(), other.size()))
        new_m = self.size(-2)
        if tensor_size_ind == -1:
            *batch_shape, m = other.shape
            output_shape = (*batch_shape, new_m)
        else:
            *batch_shape, m, n = other.shape
            output_shape = (*batch_shape, new_m, n)
        return ZeroLinearOperator(*output_shape, dtype=other.dtype, device=other.device)

    def mul(self, other: Union[float, Tensor, LinearOperator]) -> LinearOperator:
        shape = torch.broadcast_shapes(self.shape, other.shape)
        return self.__class__(*shape, dtype=self._dtype, device=self._device)

    def solve(self, right_tensor: Tensor, left_tensor: Optional[Tensor] = None) -> Tensor:
        raise RuntimeError("ZeroLinearOperators are not invertible!")

    @cached
    def to_dense(self) -> Tensor:
        return torch.zeros(*self.sizes)

    def transpose(self, dim1: int, dim2: int) -> LinearOperator:
        sizes = self.sizes.copy()
        tmp = sizes[dim1]
        sizes[dim1] = sizes[dim2]
        sizes[dim2] = tmp

        return ZeroLinearOperator(*sizes)

    def __add__(self, other: Union[Tensor, LinearOperator, float]) -> LinearOperator:
        return other
