#!/usr/bin/env python3

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from linear_operator.operators._linear_operator import IndexType, LinearOperator, to_dense


class DenseLinearOperator(LinearOperator):
    def _check_args(self, tsr):
        if not torch.is_tensor(tsr):
            return "DenseLinearOperator must take a torch.Tensor; got {}".format(tsr.__class__.__name__)
        if tsr.dim() < 2:
            return "DenseLinearOperator expects a matrix (or batches of matrices) - got a Tensor of size {}.".format(
                tsr.shape
            )

    def __init__(self, tsr):
        """
        Not a lazy tensor

        Args:
        - tsr (Tensor: matrix) a Tensor
        """
        super().__init__(tsr)
        self.tensor = tsr

    def _cholesky_solve(
        self: LinearOperator,  # shape: (*batch, N, N)
        rhs: Union[LinearOperator, Tensor],  # shape: (*batch2, N, M)
        upper: Optional[bool] = False,
    ) -> Union[LinearOperator, Tensor]:  # shape: (..., N, M)
        return torch.cholesky_solve(rhs, self.to_dense(), upper=upper)

    def _diagonal(
        self: LinearOperator,  # shape: (..., M, N)
    ) -> torch.Tensor:  # shape: (..., N)
        return self.tensor.diagonal(dim1=-1, dim2=-2)

    def _expand_batch(
        self: LinearOperator, batch_shape: Union[torch.Size, List[int]]  # shape: (..., M, N)
    ) -> LinearOperator:  # shape: (..., M, N)
        return self.__class__(self.tensor.expand(*batch_shape, *self.matrix_shape))

    def _get_indices(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> torch.Tensor:
        # Perform the __getitem__
        res = self.tensor[(*batch_indices, row_index, col_index)]
        return res

    def _getitem(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> LinearOperator:
        # Perform the __getitem__
        res = self.tensor[(*batch_indices, row_index, col_index)]
        return self.__class__(res)

    def _isclose(self, other, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> Tensor:
        return torch.isclose(self.tensor, to_dense(other), rtol=rtol, atol=atol, equal_nan=equal_nan)

    def _matmul(
        self: LinearOperator,  # shape: (*batch, M, N)
        rhs: torch.Tensor,  # shape: (*batch2, N, C) or (*batch2, N)
    ) -> torch.Tensor:  # shape: (..., M, C) or (..., M)
        return torch.matmul(self.tensor, rhs)

    def _prod_batch(self, dim: int) -> LinearOperator:
        return self.__class__(self.tensor.prod(dim))

    def _bilinear_derivative(self, left_vecs: Tensor, right_vecs: Tensor) -> Tuple[Optional[Tensor], ...]:
        res = left_vecs.matmul(right_vecs.mT)
        return (res,)

    def _size(self) -> torch.Size:
        return self.tensor.size()

    def _sum_batch(self, dim: int) -> LinearOperator:
        return self.__class__(self.tensor.sum(dim))

    def _transpose_nonbatch(
        self: LinearOperator,  # shape: (*batch, M, N)
    ) -> LinearOperator:  # shape: (*batch, N, M)
        return DenseLinearOperator(self.tensor.mT)

    def _t_matmul(
        self: LinearOperator,  # shape: (*batch, M, N)
        rhs: Union[Tensor, LinearOperator],  # shape: (*batch2, M, P)
    ) -> Union[LinearOperator, Tensor]:  # shape: (..., N, P)
        return torch.matmul(self.tensor.mT, rhs)

    def to_dense(
        self: LinearOperator,  # shape: (*batch, M, N)
    ) -> Tensor:  # shape: (*batch, M, N)
        return self.tensor

    def __add__(
        self: LinearOperator,  # shape: (..., #M, #N)
        other: Union[Tensor, LinearOperator, float],  # shape: (..., #M, #N)
    ) -> Union[LinearOperator, Tensor]:  # shape: (..., M, N)
        if isinstance(other, DenseLinearOperator):
            return DenseLinearOperator(self.tensor + other.tensor)
        elif isinstance(other, torch.Tensor):
            return DenseLinearOperator(self.tensor + other)
        else:
            return super().__add__(other)


def to_linear_operator(obj: Union[torch.Tensor, LinearOperator]) -> LinearOperator:
    """
    A function which ensures that `obj` is a LinearOperator.
    - If `obj` is a LinearOperator, this function does nothing.
    - If `obj` is a (normal) Tensor, this function wraps it with a `DenseLinearOperator`.
    """

    if torch.is_tensor(obj):
        return DenseLinearOperator(obj)
    elif isinstance(obj, LinearOperator):
        return obj
    else:
        raise TypeError("object of class {} cannot be made into a LinearOperator".format(obj.__class__.__name__))


__all__ = ["DenseLinearOperator", "to_linear_operator"]
