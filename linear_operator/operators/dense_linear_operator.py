#!/usr/bin/env python3

from __future__ import annotations

from typing import Tuple, Union

import torch
from jaxtyping import Float
from torch import Tensor

from ._linear_operator import LinearOperator


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

    def _cholesky_solve(self, rhs, upper=False):
        return torch.cholesky_solve(rhs, self.to_dense(), upper=upper)

    def _diagonal(self: Float[DenseLinearOperator, "*batch N N"]) -> Float[Tensor, "*batch N"]:
        return self.tensor.diagonal(dim1=-1, dim2=-2)

    def _expand_batch(
        self: Float[DenseLinearOperator, "... M N"], batch_shape: torch.Size
    ) -> Float[DenseLinearOperator, "... M N"]:
        return self.__class__(self.tensor.expand(*batch_shape, *self.matrix_shape))

    def _get_indices(
        self, row_index: torch.LongTensor, col_index: torch.LongTensor, *batch_indices: Tuple[torch.LongTensor, ...]
    ) -> torch.Tensor:
        # Perform the __getitem__
        res = self.tensor[(*batch_indices, row_index, col_index)]
        return res

    def _getitem(
        self,
        row_index: Union[slice, torch.LongTensor],
        col_index: Union[slice, torch.LongTensor],
        *batch_indices: Tuple[Union[int, slice, torch.LongTensor], ...],
    ) -> DenseLinearOperator:
        # Perform the __getitem__
        res = self.tensor[(*batch_indices, row_index, col_index)]
        return self.__class__(res)

    def _matmul(
        self: Float[DenseLinearOperator, "... M N"], rhs: Union[Float[Tensor, " N"], Float[Tensor, "... N P"]]
    ) -> Union[Float[Tensor, "... M P"], Float[Tensor, "... M"]]:
        return torch.matmul(self.tensor, rhs)

    def _prod_batch(self, dim):
        return self.__class__(self.tensor.prod(dim))

    def _bilinear_derivative(self, left_vecs: torch.Tensor, right_vecs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        res = left_vecs.matmul(right_vecs.mT)
        return (res,)

    def _size(self):
        return self.tensor.size()

    def _sum_batch(self, dim):
        return self.__class__(self.tensor.sum(dim))

    def _transpose_nonbatch(self: Float[DenseLinearOperator, "*batch M N"]) -> Float[DenseLinearOperator, "*batch N M"]:
        return DenseLinearOperator(self.tensor.mT)

    def _t_matmul(
        self: Float[DenseLinearOperator, "*batch M N"], rhs: Float[Tensor, "... M P"]
    ) -> Float[Tensor, "*batch N P"]:
        return torch.matmul(self.tensor.mT, rhs)

    def to_dense(self):
        return self.tensor

    def __add__(
        self: Float[DenseLinearOperator, "... M N"], other: Union[Tensor, LinearOperator, float]
    ) -> Float[LinearOperator, "... M N"]:
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
