#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from jaxtyping import Float
from torch import Tensor

from ..utils.memoize import cached
from ._linear_operator import IndexType, LinearOperator
from .root_linear_operator import RootLinearOperator


class ConstantMulLinearOperator(LinearOperator):
    """
    A LinearOperator that multiplies a base LinearOperator by a scalar constant:

    ```
    constant_mul_linear_op = constant * base_linear_op
    ```

    .. note::

        To element-wise multiply two lazy tensors, see :class:`linear_operator.operators.MulLinearOperator`

    Args:
        base_linear_op (LinearOperator) or (b x n x m)): The base_lazy tensor
        constant (Tensor): The constant

    If `base_linear_op` represents a matrix (non-batch), then `constant` must be a
    0D tensor, or a 1D tensor with one element.

    If `base_linear_op` represents a batch of matrices (b x m x n), then `constant` can be
    either:
    - A 0D tensor - the same constant is applied to all matrices in the batch
    - A 1D tensor with one element - the same constant is applied to all matrices
    - A 1D tensor with `b` elements - a different constant is applied to each matrix

    Example::

        >>> base_base_linear_op = linear_operator.operators.ToeplitzLinearOperator([1, 2, 3])
        >>> constant = torch.tensor(1.2)
        >>> new_base_linear_op = linear_operator.operators.ConstantMulLinearOperator(base_base_linear_op, constant)
        >>> new_base_linear_op.to_dense()
        >>> # Returns:
        >>> # [[ 1.2, 2.4, 3.6 ]
        >>> #  [ 2.4, 1.2, 2.4 ]
        >>> #  [ 3.6, 2.4, 1.2 ]]
        >>>
        >>> base_base_linear_op = linear_operator.operators.ToeplitzLinearOperator([[1, 2, 3], [2, 3, 4]])
        >>> constant = torch.tensor([1.2, 0.5])
        >>> new_base_linear_op = linear_operator.operators.ConstantMulLinearOperator(base_base_linear_op, constant)
        >>> new_base_linear_op.to_dense()
        >>> # Returns:
        >>> # [[[ 1.2, 2.4, 3.6 ]
        >>> #   [ 2.4, 1.2, 2.4 ]
        >>> #   [ 3.6, 2.4, 1.2 ]]
        >>> #  [[ 1, 1.5, 2 ]
        >>> #   [ 1.5, 1, 1.5 ]
        >>> #   [ 2, 1.5, 1 ]]]
    """

    def __init__(self, base_linear_op, constant):
        if not torch.is_tensor(constant):
            constant = torch.tensor(constant, device=base_linear_op.device, dtype=base_linear_op.dtype)

        super(ConstantMulLinearOperator, self).__init__(base_linear_op, constant)
        self.base_linear_op = base_linear_op
        self._constant = constant

    def _approx_diagonal(self: Float[LinearOperator, "*batch N N"]) -> Float[torch.Tensor, "*batch N"]:
        res = self.base_linear_op._approx_diagonal()
        return res * self._constant.unsqueeze(-1)

    def _diagonal(self: Float[LinearOperator, "*batch N N"]) -> Float[torch.Tensor, "... N"]:
        res = self.base_linear_op._diagonal()
        return res * self._constant.unsqueeze(-1)

    def _expand_batch(
        self: Float[ConstantMulLinearOperator, "... M N"], batch_shape: torch.Size
    ) -> Float[ConstantMulLinearOperator, "... M N"]:
        return self.__class__(
            self.base_linear_op._expand_batch(batch_shape),
            self._constant.expand(*batch_shape) if len(batch_shape) else self._constant,
        )

    def _get_indices(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> torch.Tensor:
        # NOTE TO FUTURE SELF:
        # This custom __getitem__ is actually very important!
        # It prevents constructing an InterpolatedLinearOperator when one isn't needed
        # This affects runntimes by up to 5x on simple exact GPs
        # Run __getitem__ on the base_linear_op and the constant
        base_linear_op = self.base_linear_op._get_indices(row_index, col_index, *batch_indices)
        constant = self._constant.expand(self.batch_shape)[batch_indices]
        return base_linear_op * constant

    def _getitem(
        self,
        row_index: IndexType,
        col_index: IndexType,
        *batch_indices: IndexType,
    ) -> LinearOperator:
        # NOTE TO FUTURE SELF:
        # This custom __getitem__ is actually very important!
        # It prevents constructing an InterpolatedLinearOperator when one isn't needed
        # This affects runntimes by up to 5x on simple exact GPs
        # Run __getitem__ on the base_linear_op and the constant
        base_linear_op = self.base_linear_op._getitem(row_index, col_index, *batch_indices)
        constant = self._constant.expand(self.batch_shape)[batch_indices]
        constant = constant.view(*constant.shape, 1, 1)
        return base_linear_op * constant

    def _matmul(
        self: Float[LinearOperator, "*batch M N"],
        rhs: Union[Float[torch.Tensor, "*batch2 N C"], Float[torch.Tensor, "*batch2 N"]],
    ) -> Union[Float[torch.Tensor, "... M C"], Float[torch.Tensor, "... M"]]:
        res = self.base_linear_op._matmul(rhs)
        res = res * self.expanded_constant
        return res

    def _permute_batch(self, *dims: int) -> ConstantMulLinearOperator:
        return self.__class__(
            self.base_linear_op._permute_batch(*dims), self._constant.expand(self.batch_shape).permute(*dims)
        )

    def _bilinear_derivative(self, left_vecs: Tensor, right_vecs: Tensor) -> Tuple[Optional[Tensor], ...]:
        # Gradient with respect to the constant
        constant_deriv = left_vecs * self.base_linear_op._matmul(right_vecs)
        constant_deriv = constant_deriv.sum(-2).sum(-1)
        while constant_deriv.dim() > self._constant.dim():
            constant_deriv = constant_deriv.sum(0)
        for i in range(self._constant.dim()):
            if self._constant.size(i) == 1:
                constant_deriv = constant_deriv.sum(i, keepdim=True)

        # Get derivaties of everything else
        left_vecs = left_vecs * self.expanded_constant
        res = self.base_linear_op._bilinear_derivative(left_vecs, right_vecs)

        return tuple(res) + (constant_deriv,)

    def _size(self) -> torch.Size:
        return self.base_linear_op.size()

    def _t_matmul(
        self: Float[LinearOperator, "*batch M N"],
        rhs: Union[Float[Tensor, "*batch2 M P"], Float[LinearOperator, "*batch2 M P"]],
    ) -> Union[Float[LinearOperator, "... N P"], Float[Tensor, "... N P"]]:
        res = self.base_linear_op._t_matmul(rhs)
        res = res * self.expanded_constant
        return res

    def _transpose_nonbatch(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch N M"]:
        return ConstantMulLinearOperator(self.base_linear_op._transpose_nonbatch(), self._constant)

    def _unsqueeze_batch(self, dim: int) -> ConstantMulLinearOperator:
        broadcasted_shape = self.batch_shape
        base_linear_op = self.base_linear_op._expand_batch(broadcasted_shape)._unsqueeze_batch(dim)
        constant = self._constant.expand(broadcasted_shape).unsqueeze(dim)
        return ConstantMulLinearOperator(base_linear_op=base_linear_op, constant=constant)

    @property
    def expanded_constant(self) -> Tensor:
        # Make sure that the constant can be expanded to the appropriate size
        try:
            constant = self._constant.view(*self._constant.shape, 1, 1)
        except RuntimeError:
            raise RuntimeError(
                "ConstantMulLinearOperator of size {} received an invalid constant of size {}.".format(
                    self.base_linear_op.shape, self._constant.shape
                )
            )

        return constant

    @cached
    def to_dense(self):
        res = self.base_linear_op.to_dense()
        return res * self.expanded_constant

    @cached(name="root_decomposition")
    def root_decomposition(
        self: Float[LinearOperator, "*batch N N"], method: Optional[str] = None
    ) -> Float[LinearOperator, "*batch N N"]:
        if torch.all(self._constant >= 0):
            base_root = self.base_linear_op.root_decomposition(method=method).root
            return RootLinearOperator(ConstantMulLinearOperator(base_root, self._constant**0.5))

        return super().root_decomposition(method=method)
