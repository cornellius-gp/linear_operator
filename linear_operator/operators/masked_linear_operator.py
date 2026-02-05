from __future__ import annotations

import torch
from torch import Tensor

from linear_operator.operators._linear_operator import _is_noop_index, IndexType, LinearOperator

from linear_operator.utils.generic import _to_helper


class MaskedLinearOperator(LinearOperator):
    r"""
    A :obj:`~linear_operator.operators.LinearOperator` that applies a mask to the rows and columns of a base
    :obj:`~linear_operator.operators.LinearOperator`.
    """

    def __init__(
        self,
        base: LinearOperator,  # shape: (*batch, M0, N0)
        row_mask: Tensor,  # shape: (M0)
        col_mask: Tensor,  # shape: (N0)
    ):
        r"""
        Create a new :obj:`~linear_operator.operators.MaskedLinearOperator` that applies a mask to the rows and columns
        of a base :obj:`~linear_operator.operators.LinearOperator`.

        :param base: The base :obj:`~linear_operator.operators.LinearOperator`.
        :param row_mask: A :obj:`torch.BoolTensor` containing the mask to apply to the rows.
        :param col_mask: A :obj:`torch.BoolTensor` containing the mask to apply to the columns.
        """
        super().__init__(base, row_mask, col_mask)
        self.base = base
        self.row_mask = row_mask
        self.col_mask = col_mask
        self.row_eq_col_mask = torch.equal(row_mask, col_mask)

    @staticmethod
    def _expand(
        tensor: Tensor,  # shape: (*batch, N, C)
        mask: Tensor,  # shape: (N0)
    ) -> Tensor:  # shape: (*batch, N0, C)
        res = torch.zeros(
            *tensor.shape[:-2],
            mask.size(-1),
            tensor.size(-1),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        res[..., mask, :] = tensor
        return res

    def _matmul(
        self: LinearOperator,  # shape: (*batch, M, N)
        rhs: torch.Tensor,  # shape: (*batch2, N, C) or (*batch2, N)
    ) -> torch.Tensor:  # shape: (..., M, C) or (..., M)
        rhs_expanded = self._expand(rhs, self.col_mask)
        res_expanded = self.base._matmul(rhs_expanded)
        res = res_expanded[..., self.row_mask, :]

        return res

    def _t_matmul(
        self: LinearOperator,  # shape: (*batch, M, N)
        rhs: Tensor | LinearOperator,  # shape: (*batch2, M, P)
    ) -> LinearOperator | Tensor:  # shape: (..., N, P)
        rhs_expanded = self._expand(rhs, self.row_mask)
        res_expanded = self.base._t_matmul(rhs_expanded)
        res = res_expanded[..., self.col_mask, :]
        return res

    def _size(self) -> torch.Size:
        return torch.Size(
            (*self.base.size()[:-2], torch.count_nonzero(self.row_mask), torch.count_nonzero(self.col_mask))
        )

    def _transpose_nonbatch(
        self: LinearOperator,  # shape: (*batch, M, N)
    ) -> LinearOperator:  # shape: (*batch, N, M)
        return self.__class__(self.base.mT, self.col_mask, self.row_mask)

    def _diagonal(
        self: LinearOperator,  # shape: (..., M, N)
    ) -> torch.Tensor:  # shape: (..., N)
        if not self.row_eq_col_mask:
            raise NotImplementedError()
        diag = self.base.diagonal()
        return diag[..., self.row_mask]

    def to_dense(
        self: LinearOperator,  # shape: (*batch, M, N)
    ) -> Tensor:  # shape: (*batch, M, N)
        full_dense = self.base.to_dense()
        return full_dense[..., self.row_mask, :][..., :, self.col_mask]

    def _bilinear_derivative(self, left_vecs: Tensor, right_vecs: Tensor) -> tuple[Tensor | None, ...]:
        left_vecs = self._expand(left_vecs, self.row_mask)
        right_vecs = self._expand(right_vecs, self.col_mask)
        return self.base._bilinear_derivative(left_vecs, right_vecs) + (None, None)

    def _expand_batch(
        self: LinearOperator, batch_shape: torch.Size | list[int]  # shape: (..., M, N)
    ) -> LinearOperator:  # shape: (..., M, N)
        return self.__class__(self.base._expand_batch(batch_shape), self.row_mask, self.col_mask)

    def _unsqueeze_batch(self, dim: int) -> LinearOperator:
        return self.__class__(self.base._unsqueeze_batch(dim), self.row_mask, self.col_mask)

    def _getitem(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> LinearOperator:
        if _is_noop_index(row_index) and _is_noop_index(col_index):
            if len(batch_indices):
                return self.__class__(self.base[batch_indices], self.row_mask, self.col_mask)
            else:
                return self
        else:
            return super()._getitem(row_index, col_index, *batch_indices)

    def _get_indices(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> torch.Tensor:
        row_mapping = torch.arange(self.base.size(-2), device=self.base.device)[self.row_mask]
        col_mapping = torch.arange(self.base.size(-1), device=self.base.device)[self.col_mask]
        return self.base._get_indices(row_mapping[row_index], col_mapping[col_index], *batch_indices)

    def _permute_batch(self, *dims: int) -> LinearOperator:
        return self.__class__(self.base._permute_batch(*dims), self.row_mask, self.col_mask)

    def to(
        self: LinearOperator,  # shape: (*batch, M, N)
        *args,
        **kwargs,
    ) -> LinearOperator:  # shape: (*batch, M, N)

        # Overwrite the to() method in _linear_operator to avoid converting mask matrices to float.
        # Will only convert both dtype and device when arg's dtype is not torch.bool.
        # Otherwise, will only convert device.

        device, dtype = _to_helper(*args, **kwargs)

        new_args = []
        new_kwargs = {}
        for arg in self._args:
            if hasattr(arg, "to"):
                if hasattr(arg, "dtype") and arg.dtype.is_floating_point == dtype.is_floating_point:
                    new_args.append(arg.to(dtype=dtype, device=device))
                else:
                    new_args.append(arg.to(device=device))
            else:
                new_args.append(arg)
        for name, val in self._kwargs.items():
            if hasattr(val, "to"):
                new_kwargs[name] = val.to(dtype=dtype, device=device)
            else:
                new_kwargs[name] = val
        return self.__class__(*new_args, **new_kwargs)
