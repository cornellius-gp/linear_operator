#!/usr/bin/env python3

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
from jaxtyping import Float
from torch import Tensor

from ..utils.generic import _to_helper

from ..utils.getitem import _compute_getitem_size, _is_noop_index
from ..utils.memoize import cached
from ._linear_operator import IndexType, LinearOperator
from .diag_linear_operator import ConstantDiagLinearOperator
from .zero_linear_operator import ZeroLinearOperator


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
        self: Float[LinearOperator, "*batch N N"], upper: Optional[bool] = False
    ) -> Float[LinearOperator, "*batch N N"]:
        return self

    def _cholesky_solve(
        self: Float[LinearOperator, "*batch N N"],
        rhs: Union[Float[LinearOperator, "*batch2 N M"], Float[Tensor, "*batch2 N M"]],
        upper: Optional[bool] = False,
    ) -> Union[Float[LinearOperator, "... N M"], Float[Tensor, "... N M"]]:
        return self._maybe_reshape_rhs(rhs)

    def _expand_batch(
        self: Float[LinearOperator, "... M N"], batch_shape: Union[torch.Size, List[int]]
    ) -> Float[LinearOperator, "... M N"]:
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
        self: Float[LinearOperator, "*batch M N"],
        rhs: Union[Float[torch.Tensor, "*batch2 N C"], Float[torch.Tensor, "*batch2 N"]],
    ) -> Union[Float[torch.Tensor, "... M C"], Float[torch.Tensor, "... M"]]:
        return self._maybe_reshape_rhs(rhs)

    def _mul_constant(
        self: Float[LinearOperator, "*batch M N"], other: Union[float, torch.Tensor]
    ) -> Float[LinearOperator, "*batch M N"]:
        return ConstantDiagLinearOperator(self.diag_values * other, diag_shape=self.diag_shape)

    def _mul_matrix(
        self: Float[LinearOperator, "... #M #N"],
        other: Union[Float[torch.Tensor, "... #M #N"], Float[LinearOperator, "... #M #N"]],
    ) -> Float[LinearOperator, "... M N"]:
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
        self: Float[LinearOperator, "... N N"]
    ) -> Union[Float[torch.Tensor, "... N N"], Float[LinearOperator, "... N N"]]:
        return self.sqrt()

    def _root_inv_decomposition(
        self: Float[LinearOperator, "*batch N N"],
        initial_vectors: Optional[torch.Tensor] = None,
        test_vectors: Optional[torch.Tensor] = None,
    ) -> Union[Float[LinearOperator, "... N N"], Float[Tensor, "... N N"]]:
        return self.inverse().sqrt()

    def _size(self) -> torch.Size:
        return torch.Size([*self._batch_shape, self.diag_shape, self.diag_shape])

    @cached(name="svd")
    def _svd(
        self: Float[LinearOperator, "*batch N N"]
    ) -> Tuple[Float[LinearOperator, "*batch N N"], Float[Tensor, "... N"], Float[LinearOperator, "*batch N N"]]:
        return self, self._diag, self

    def _symeig(
        self: Float[LinearOperator, "*batch N N"],
        eigenvectors: bool = False,
        return_evals_as_lazy: Optional[bool] = False,
    ) -> Tuple[Float[Tensor, "*batch M"], Optional[Float[LinearOperator, "*batch N M"]]]:
        return self._diag, self

    def _t_matmul(
        self: Float[LinearOperator, "*batch M N"],
        rhs: Union[Float[Tensor, "*batch2 M P"], Float[LinearOperator, "*batch2 M P"]],
    ) -> Union[Float[LinearOperator, "... N P"], Float[Tensor, "... N P"]]:
        return self._maybe_reshape_rhs(rhs)

    def _transpose_nonbatch(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch N M"]:
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

    def exp(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch M N"]:
        return self

    def inverse(self: Float[LinearOperator, "*batch N N"]) -> Float[LinearOperator, "*batch N N"]:
        return self

    def inv_quad_logdet(
        self: Float[LinearOperator, "*batch N N"],
        inv_quad_rhs: Optional[Union[Float[Tensor, "*batch N M"], Float[Tensor, "*batch N"]]] = None,
        logdet: Optional[bool] = False,
        reduce_inv_quad: Optional[bool] = True,
    ) -> Tuple[
        Optional[Union[Float[Tensor, "*batch M"], Float[Tensor, " *batch"], Float[Tensor, " 0"]]],
        Optional[Float[Tensor, "..."]],
    ]:
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

    def log(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch M N"]:
        return ZeroLinearOperator(
            *self._batch_shape, self.diag_shape, self.diag_shape, dtype=self._dtype, device=self._device
        )

    def matmul(
        self: Float[LinearOperator, "*batch M N"],
        other: Union[Float[Tensor, "*batch2 N P"], Float[Tensor, "*batch2 N"], Float[LinearOperator, "*batch2 N P"]],
    ) -> Union[Float[Tensor, "... M P"], Float[Tensor, "... M"], Float[LinearOperator, "... M P"]]:
        is_vec = False
        if other.dim() == 1:
            is_vec = True
            other = other.unsqueeze(-1)
        res = self._maybe_reshape_rhs(other)
        if is_vec:
            res = res.squeeze(-1)
        return res

    def solve(
        self: Float[LinearOperator, "... N N"],
        right_tensor: Union[Float[Tensor, "... N P"], Float[Tensor, " N"]],
        left_tensor: Optional[Float[Tensor, "... O N"]] = None,
    ) -> Union[Float[Tensor, "... N P"], Float[Tensor, "... N"], Float[Tensor, "... O P"], Float[Tensor, "... O"]]:
        res = self._maybe_reshape_rhs(right_tensor)
        if left_tensor is not None:
            res = left_tensor @ res
        return res

    def sqrt(self: Float[LinearOperator, "*batch M N"]) -> Float[LinearOperator, "*batch M N"]:
        return self

    def sqrt_inv_matmul(
        self: Float[LinearOperator, "*batch N N"],
        rhs: Float[Tensor, "*batch N P"],
        lhs: Optional[Float[Tensor, "*batch O N"]] = None,
    ) -> Union[Float[Tensor, "*batch N P"], Tuple[Float[Tensor, "*batch O P"], Float[Tensor, "*batch O"]]]:
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
        self: Float[LinearOperator, "*batch N N"], num_samples: int
    ) -> Float[Tensor, "num_samples *batch N"]:
        base_samples = torch.randn(num_samples, *self.shape[:-1], dtype=self.dtype, device=self.device)
        return base_samples

    def to(self: Float[LinearOperator, "*batch M N"], *args, **kwargs) -> Float[LinearOperator, "*batch M N"]:

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
