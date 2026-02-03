#!/usr/bin/env python3

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from linear_operator.operators._linear_operator import IndexType, LinearOperator
from linear_operator.operators.diag_linear_operator import ConstantDiagLinearOperator
from linear_operator.operators.zero_linear_operator import ZeroLinearOperator

from linear_operator.utils.generic import _to_helper
from linear_operator.utils.getitem import _compute_getitem_size, _is_noop_index
from linear_operator.utils.memoize import cached


class IdentityLinearOperator(ConstantDiagLinearOperator):
    """
    Identity linear operator. Supports arbitrary batch sizes.

    :param diag_shape: The size of the identity matrix (i.e. :math:`N`).
    :param batch_shape: The size of the batch dimensions. It may be useful to set these dimensions for broadcasting.
    :param dtype: Dtype that the LinearOperator will be operating on. (Default: :meth:`torch.get_default_dtype()`).
    :param device: Device that the LinearOperator will be operating on. (Default: CPU).
    """

    def __init__(
        self,
        diag_shape: int,
        batch_shape: Optional[torch.Size] = torch.Size([]),
        dtype: Optional[torch.dtype] = torch.float,
        device: Optional[torch.device] = None,
    ):
        one = torch.tensor(1.0, dtype=dtype, device=device)
        LinearOperator.__init__(self, diag_shape=diag_shape, batch_shape=batch_shape, dtype=dtype, device=device)
        self.diag_values = one.expand(torch.Size([*batch_shape, 1]))
        self.diag_shape = diag_shape
        self._batch_shape = batch_shape
        self._dtype = dtype
        self._device = device

    @property
    def batch_shape(self) -> torch.Size:
        return self._batch_shape

    @property
    def dtype(self) -> Optional[torch.dtype]:
        return self._dtype

    @property
    def device(self) -> Optional[torch.device]:
        return self._device

    def _maybe_reshape_rhs(self, rhs: Union[torch.Tensor, LinearOperator]) -> Union[torch.Tensor, LinearOperator]:
        if self._batch_shape != rhs.shape[:-2]:
            batch_shape = torch.broadcast_shapes(rhs.shape[:-2], self._batch_shape)
            return rhs.expand(*batch_shape, *rhs.shape[-2:])
        else:
            return rhs

    @cached(name="cholesky", ignore_args=True)
    def _cholesky(
        self: LinearOperator, upper: Optional[bool] = False  # shape: (*batch, N, N)
    ) -> LinearOperator:  # shape: (*batch, N, N)
        return self

    def _cholesky_solve(
        self: LinearOperator,  # shape: (*batch, N, N)
        rhs: Union[LinearOperator, Tensor],  # shape: (*batch2, N, M)
        upper: Optional[bool] = False,
    ) -> Union[LinearOperator, Tensor]:  # shape: (..., N, M)
        return self._maybe_reshape_rhs(rhs)

    def _expand_batch(
        self: LinearOperator, batch_shape: Union[torch.Size, List[int]]  # shape: (..., M, N)
    ) -> LinearOperator:  # shape: (..., M, N)
        return IdentityLinearOperator(
            diag_shape=self.diag_shape, batch_shape=batch_shape, dtype=self.dtype, device=self.device
        )

    def _getitem(self, row_index: IndexType, col_index: IndexType, *batch_indices: IndexType) -> LinearOperator:
        # Special case: if both row and col are not indexed, then we are done
        if _is_noop_index(row_index) and _is_noop_index(col_index):
            if len(batch_indices):
                new_batch_shape = _compute_getitem_size(self, (*batch_indices, row_index, col_index))[:-2]
                res = IdentityLinearOperator(
                    diag_shape=self.diag_shape, batch_shape=new_batch_shape, dtype=self._dtype, device=self._device
                )
                return res
            else:
                return self

        return super()._getitem(row_index, col_index, *batch_indices)

    def _matmul(
        self: LinearOperator,  # shape: (*batch, M, N)
        rhs: torch.Tensor,  # shape: (*batch2, N, C) or (*batch2, N)
    ) -> torch.Tensor:  # shape: (..., M, C) or (..., M)
        return self._maybe_reshape_rhs(rhs)

    def _mul_constant(
        self: LinearOperator, other: Union[float, torch.Tensor]  # shape: (*batch, M, N)
    ) -> LinearOperator:  # shape: (*batch, M, N)
        return ConstantDiagLinearOperator(self.diag_values * other, diag_shape=self.diag_shape)

    def _mul_matrix(
        self: LinearOperator,  # shape: (..., #M, #N)
        other: Union[torch.Tensor, LinearOperator],  # shape: (..., #M, #N)
    ) -> LinearOperator:  # shape: (..., M, N)
        return other

    def _permute_batch(self, *dims: int) -> LinearOperator:
        batch_shape = self.diag_values.permute(*dims, -1).shape[:-1]
        return IdentityLinearOperator(
            diag_shape=self.diag_shape, batch_shape=batch_shape, dtype=self._dtype, device=self._device
        )

    def _prod_batch(self, dim: int) -> LinearOperator:
        batch_shape = list(self.batch_shape)
        del batch_shape[dim]
        return IdentityLinearOperator(
            diag_shape=self.diag_shape, batch_shape=torch.Size(batch_shape), dtype=self.dtype, device=self.device
        )

    def _root_decomposition(
        self: LinearOperator,  # shape: (..., N, N)
    ) -> Union[torch.Tensor, LinearOperator]:  # shape: (..., N, N)
        return self.sqrt()

    def _root_inv_decomposition(
        self: LinearOperator,  # shape: (*batch, N, N)
        initial_vectors: Optional[torch.Tensor] = None,
        test_vectors: Optional[torch.Tensor] = None,
    ) -> Union[LinearOperator, Tensor]:  # shape: (..., N, N)
        return self.inverse().sqrt()

    def _size(self) -> torch.Size:
        return torch.Size([*self._batch_shape, self.diag_shape, self.diag_shape])

    @cached(name="svd")
    def _svd(
        self: LinearOperator,  # shape: (*batch, N, N)
    ) -> Tuple[LinearOperator, Tensor, LinearOperator]:  # shape: (*batch, N, N), (..., N), (*batch, N, N)
        return self, self._diag, self

    def _symeig(
        self: LinearOperator,  # shape: (*batch, N, N)
        eigenvectors: bool = False,
        return_evals_as_lazy: Optional[bool] = False,
    ) -> Tuple[Tensor, Optional[LinearOperator]]:  # shape: (*batch, M), (*batch, N, M)
        return self._diag, self

    def _t_matmul(
        self: LinearOperator,  # shape: (*batch, M, N)
        rhs: Union[Tensor, LinearOperator],  # shape: (*batch2, M, P)
    ) -> Union[LinearOperator, Tensor]:  # shape: (..., N, P)
        return self._maybe_reshape_rhs(rhs)

    def _transpose_nonbatch(
        self: LinearOperator,  # shape: (*batch, M, N)
    ) -> LinearOperator:  # shape: (*batch, N, M)
        return self

    def _unsqueeze_batch(self, dim: int) -> LinearOperator:
        batch_shape = list(self._batch_shape)
        batch_shape.insert(dim, 1)
        batch_shape = torch.Size(batch_shape)
        return IdentityLinearOperator(
            diag_shape=self.diag_shape, batch_shape=batch_shape, dtype=self.dtype, device=self.device
        )

    def abs(self) -> LinearOperator:
        return self

    def exp(
        self: LinearOperator,  # shape: (*batch, M, N)
    ) -> LinearOperator:  # shape: (*batch, M, N)
        return self

    def inverse(
        self: LinearOperator,  # shape: (*batch, N, N)
    ) -> LinearOperator:  # shape: (*batch, N, N)
        return self

    def inv_quad_logdet(
        self: LinearOperator,  # shape: (*batch, N, N)
        inv_quad_rhs: Optional[Tensor] = None,  # shape: (*batch, N, M) or (*batch, N)
        logdet: Optional[bool] = False,
        reduce_inv_quad: Optional[bool] = True,
    ) -> Tuple[  # fmt: off
        Optional[Tensor],  # shape: (*batch, M) or (*batch) or (0)
        Optional[Tensor],  # shape: (...)
    ]:  # fmt: on
        # TODO: Use proper batching for inv_quad_rhs (prepand to shape rather than append)
        if inv_quad_rhs is None:
            inv_quad_term = torch.empty(0, dtype=self.dtype, device=self.device)
        else:
            rhs_batch_shape = inv_quad_rhs.shape[1 + self.batch_dim :]
            inv_quad_term = inv_quad_rhs.mul(inv_quad_rhs).sum(-(1 + len(rhs_batch_shape)))
            if reduce_inv_quad:
                inv_quad_term = inv_quad_term.sum(-1)

        if logdet:
            logdet_term = torch.zeros(self.batch_shape, dtype=self.dtype, device=self.device)
        else:
            logdet_term = torch.empty(0, dtype=self.dtype, device=self.device)

        return inv_quad_term, logdet_term

    def log(
        self: LinearOperator,  # shape: (*batch, M, N)
    ) -> LinearOperator:  # shape: (*batch, M, N)
        return ZeroLinearOperator(
            *self._batch_shape, self.diag_shape, self.diag_shape, dtype=self._dtype, device=self._device
        )

    def matmul(
        self: LinearOperator,  # shape: (*batch, M, N)
        other: Union[Tensor, LinearOperator],  # shape: (*batch2, N, P) or (*batch2, N)
    ) -> Union[Tensor, LinearOperator]:  # shape: (..., M, P) or (..., M)
        is_vec = False
        if other.dim() == 1:
            is_vec = True
            other = other.unsqueeze(-1)
        res = self._maybe_reshape_rhs(other)
        if is_vec:
            res = res.squeeze(-1)
        return res

    def solve(
        self: LinearOperator,  # shape: (..., N, N)
        right_tensor: Tensor,  # shape: (..., N, P) or (N)
        left_tensor: Optional[Tensor] = None,  # shape: (..., O, N)
    ) -> Tensor:  # shape: (..., N, P) or (..., N) or (..., O, P) or (..., O)
        res = self._maybe_reshape_rhs(right_tensor)
        if left_tensor is not None:
            res = left_tensor @ res
        return res

    def sqrt(
        self: LinearOperator,  # shape: (*batch, M, N)
    ) -> LinearOperator:  # shape: (*batch, M, N)
        return self

    def sqrt_inv_matmul(
        self: LinearOperator,  # shape: (*batch, N, N)
        rhs: Tensor,  # shape: (*batch, N, P)
        lhs: Optional[Tensor] = None,  # shape: (*batch, O, N)
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:  # shape: (*batch, N, P), (*batch, O, P), (*batch, O)
        if lhs is None:
            return self._maybe_reshape_rhs(rhs)
        else:
            sqrt_inv_matmul = lhs @ rhs
            inv_quad = lhs.pow(2).sum(dim=-1)
            return sqrt_inv_matmul, inv_quad

    def type(self: LinearOperator, dtype: torch.dtype) -> LinearOperator:
        return IdentityLinearOperator(
            diag_shape=self.diag_shape, batch_shape=self.batch_shape, dtype=dtype, device=self.device
        )

    def zero_mean_mvn_samples(
        self: LinearOperator, num_samples: int  # shape: (*batch, N, N)
    ) -> Tensor:  # shape: (num_samples, *batch, N)
        base_samples = torch.randn(num_samples, *self.shape[:-1], dtype=self.dtype, device=self.device)
        return base_samples

    def to(
        self: LinearOperator,  # shape: (*batch, M, N)
        *args,
        **kwargs,
    ) -> LinearOperator:  # shape: (*batch, M, N)

        # Overwrite the to() method in _linear_operator to also convert the dtype and device saved in _kwargs.

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
        new_kwargs["device"] = device
        new_kwargs["dtype"] = dtype
        return self.__class__(*new_args, **new_kwargs)
