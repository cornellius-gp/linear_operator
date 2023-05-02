#!/usr/bin/env python3
from typing import Union

import torch
from jaxtyping import Float
from torch import Tensor

from ._linear_operator import LinearOperator
from .root_linear_operator import RootLinearOperator


class LowRankRootLinearOperator(RootLinearOperator):
    """
    Very thin wrapper around RootLinearOperator that denotes that the tensor specifically represents a low rank
    decomposition of a full rank matrix.

    The rationale for this class existing is that we can create LowRankAddedDiagLinearOperator without having to
    write custom _getitem, _get_indices, etc, leading to much better code reuse.
    """

    def add_diagonal(
        self: Float[LinearOperator, "*batch N N"],
        diag: Union[Float[torch.Tensor, "... N"], Float[torch.Tensor, "... 1"], Float[torch.Tensor, ""]],
    ) -> Float[LinearOperator, "*batch N N"]:
        from .diag_linear_operator import ConstantDiagLinearOperator, DiagLinearOperator
        from .low_rank_root_added_diag_linear_operator import LowRankRootAddedDiagLinearOperator

        if not self.is_square:
            raise RuntimeError("add_diag only defined for square matrices")

        diag_shape = diag.shape
        if len(diag_shape) == 0:
            # interpret scalar tensor as constant diag
            diag_tensor = ConstantDiagLinearOperator(diag.unsqueeze(-1), diag_shape=self.shape[-1])
        elif diag_shape[-1] == 1:
            # interpret single-trailing element as constant diag
            diag_tensor = ConstantDiagLinearOperator(diag, diag_shape=self.shape[-1])
        else:
            try:
                expanded_diag = diag.expand(self.shape[:-1])
            except RuntimeError:
                raise RuntimeError(
                    "add_diag for LinearOperator of size {} received invalid diagonal of size {}.".format(
                        self.shape, diag_shape
                    )
                )
            diag_tensor = DiagLinearOperator(expanded_diag)

        return LowRankRootAddedDiagLinearOperator(self, diag_tensor)

    def __add__(
        self: Float[LinearOperator, "... #M #N"],
        other: Union[Float[Tensor, "... #M #N"], Float[LinearOperator, "... #M #N"], float],
    ) -> Union[Float[LinearOperator, "... M N"], Float[Tensor, "... M N"]]:
        from .diag_linear_operator import DiagLinearOperator
        from .low_rank_root_added_diag_linear_operator import LowRankRootAddedDiagLinearOperator

        if isinstance(other, DiagLinearOperator):
            return LowRankRootAddedDiagLinearOperator(self, other)
        else:
            return super().__add__(other)
