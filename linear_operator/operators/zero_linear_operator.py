#!/usr/bin/env python3

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from linear_operator.operators._linear_operator import IndexType, LinearOperator

from linear_operator.utils.getitem import _compute_getitem_size
from linear_operator.utils.memoize import cached


class ZeroLinearOperator(LinearOperator):
    """
    Special LinearOperator representing zero.

    :param sizes: The size of each dimension (including batch dimensions).
    :param dtype: Dtype that the LinearOperator will be operating on. (Default: :meth:`torch.get_default_dtype()`).
    :param device: Device that the LinearOperator will be operating on. (Default: CPU).
    """

    def __init__(
        self, *sizes: Tuple[int, ...], dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None
    ):
        super(ZeroLinearOperator, self).__init__(*sizes)
        self.sizes = list(sizes)

        self._dtype = dtype or torch.get_default_dtype()
        self._device = device or torch.device("cpu")

    @property
    def dtype(self) -> Optional[torch.dtype]:
        return self._dtype

    @property
    def device(self) -> Optional[torch.device]:
        return self._device

    def _bilinear_derivative(self, left_vecs: Tensor, right_vecs: Tensor) -> Tuple[Optional[Tensor], ...]:
        raise RuntimeError("Backwards through a ZeroLinearOperator is not possible")

    def _diagonal(
        self: LinearOperator,  # shape: (..., M, N)
    ) -> torch.Tensor:  # shape: (..., N)
        shape = self.shape
        return torch.zeros(shape[:-1], dtype=self.dtype, device=self.device)

    def _expand_batch(
        self: LinearOperator, batch_shape: Union[torch.Size, List[int]]  # shape: (..., M, N)
    ) -> LinearOperator:  # shape: (..., M, N)
        return self.__class__(*batch_shape, *self.sizes[-2:], dtype=self._dtype, device=self._device)

    def _get_indices(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> torch.Tensor:
        new_size = _compute_getitem_size(self, batch_indices + (row_index, col_index))
        return torch.zeros(*new_size)

    def _getitem(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> LinearOperator:
        new_size = _compute_getitem_size(self, batch_indices + (row_index, col_index))
        return ZeroLinearOperator(*new_size)

    def _matmul(
        self: LinearOperator,  # shape: (*batch, M, N)
        rhs: torch.Tensor,  # shape: (*batch2, N, C) or (*batch2, N)
    ) -> torch.Tensor:  # shape: (..., M, C) or (..., M)
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

    def _root_decomposition(
        self: LinearOperator,  # shape: (..., N, N)
    ) -> Union[torch.Tensor, LinearOperator]:  # shape: (..., N, N)
        raise RuntimeError("ZeroLinearOperators are not positive definite!")

    def _root_decomposition_size(self) -> int:
        raise RuntimeError("ZeroLinearOperators are not positive definite!")

    def _root_inv_decomposition(
        self: LinearOperator,  # shape: (*batch, N, N)
        initial_vectors: Optional[torch.Tensor] = None,
        test_vectors: Optional[torch.Tensor] = None,
    ) -> Union[LinearOperator, Tensor]:  # shape: (..., N, N)
        raise RuntimeError("ZeroLinearOperators are not positive definite!")

    def _size(self) -> torch.Size:
        return torch.Size(self.sizes)

    def _sum_batch(self, dim: int) -> LinearOperator:
        sizes = list(self.sizes)
        del sizes[dim]
        return self.__class__(*sizes, dtype=self._dtype, device=self._device)

    def _t_matmul(
        self: LinearOperator,  # shape: (*batch, M, N)
        rhs: Union[Tensor, LinearOperator],  # shape: (*batch2, M, P)
    ) -> Union[LinearOperator, Tensor]:  # shape: (..., N, P)
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

    def _transpose_nonbatch(
        self: LinearOperator,  # shape: (*batch, M, N)
    ) -> LinearOperator:  # shape: (*batch, N, M)
        return self.mT

    def _unsqueeze_batch(self, dim: int) -> LinearOperator:
        sizes = self.sizes.copy()
        sizes.insert(dim, 1)
        return self.__class__(*sizes, dtype=self._dtype, device=self._device)

    def add_diagonal(
        self: LinearOperator,  # shape: (*batch, N, N)
        diag: torch.Tensor,  # shape: (..., N) or (..., 1) or ()
    ) -> LinearOperator:  # shape: (*batch, N, N)
        from linear_operator.operators.diag_linear_operator import DiagLinearOperator

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

    def div(self, other: Union[float, torch.Tensor]) -> LinearOperator:
        return self

    def inv_quad(
        self: LinearOperator,  # shape: (*batch, N, N)
        inv_quad_rhs: Tensor,  # shape: (*batch, N, M) or (*batch, N)
        reduce_inv_quad: bool = True,
    ) -> Tensor:  # shape: (*batch, M) or (*batch)
        raise RuntimeError("ZeroLinearOperators are not invertible!")

    def inv_quad_logdet(
        self: LinearOperator,  # shape: (*batch, N, N)
        inv_quad_rhs: Optional[Tensor] = None,  # shape: (*batch, N, M) or (*batch, N)
        logdet: Optional[bool] = False,
        reduce_inv_quad: Optional[bool] = True,
    ) -> Tuple[  # fmt: off
        Optional[Tensor],  # shape: (*batch, M) or (*batch) or (0)
        Optional[Tensor],  # shape: (...)
    ]:  # fmt: on
        raise RuntimeError("ZeroLinearOperators are not invertible!")

    def logdet(
        self: LinearOperator,  # shape: (*batch, M, N)
    ) -> Tensor:  # shape: (*batch)
        return torch.log(torch.tensor(0.0))

    def matmul(
        self: LinearOperator,  # shape: (*batch, M, N)
        other: Union[Tensor, LinearOperator],  # shape: (*batch2, N, P) or (*batch2, N)
    ) -> Union[Tensor, LinearOperator]:  # shape: (..., M, P) or (..., M)
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

    def mul(
        self: LinearOperator,  # shape: (*batch, M, N)
        other: Union[float, Tensor, LinearOperator],  # shape: (*batch2, M, N)
    ) -> LinearOperator:  # shape: (..., M, N)
        shape = torch.broadcast_shapes(self.shape, other.shape)
        return self.__class__(*shape, dtype=self._dtype, device=self._device)

    def solve(
        self: LinearOperator,  # shape: (..., N, N)
        right_tensor: Tensor,  # shape: (..., N, P) or (N)
        left_tensor: Optional[Tensor] = None,  # shape: (..., O, N)
    ) -> Tensor:  # shape: (..., N, P) or (..., N) or (..., O, P) or (..., O)
        raise RuntimeError("ZeroLinearOperators are not invertible!")

    @cached
    def to_dense(
        self: LinearOperator,  # shape: (*batch, M, N)
    ) -> Tensor:  # shape: (*batch, M, N)
        return torch.zeros(*self.sizes)

    def transpose(self, dim1: int, dim2: int) -> LinearOperator:
        sizes = self.sizes.copy()
        tmp = sizes[dim1]
        sizes[dim1] = sizes[dim2]
        sizes[dim2] = tmp

        return ZeroLinearOperator(*sizes)

    def __add__(
        self: LinearOperator,  # shape: (..., #M, #N)
        other: Union[Tensor, LinearOperator, float],  # shape: (..., #M, #N)
    ) -> Union[LinearOperator, Tensor]:  # shape: (..., M, N)
        return other
