from typing import Union

import torch
from jaxtyping import Float
from torch import Tensor

from ._linear_operator import LinearOperator
from .added_diag_linear_operator import AddedDiagLinearOperator

from .diag_linear_operator import DiagLinearOperator


class BlockDiagonalSparseLinearOperator(LinearOperator):
    """A sparse linear operator (which when reordered) has dense blocks on its diagonal.

    Linear operator with a matrix representation that has sparse rows, with an equal number of
    non-zero entries per row. The non-zero entries are stored in a tensor of size M x NNZ, where M is
    the number of rows and NNZ is the number of non-zero entries per row. When appropriately re-ordering
    the columns of the matrix, it is a block-diagonal matrix.

    Note:
        This currently only supports equally sized blocks of size 1 x NNZ.

    :param non_zero_idcs: Tensor of non-zero indices.
    :param blocks: Tensor of non-zero entries.
    :param size_input_dim: Size of the (sparse) input dimension, equivalently the number of columns.
    """

    def __init__(
        self,
        non_zero_idcs: Float[torch.Tensor, "M NNZ"],
        blocks: Float[torch.Tensor, "M NNZ"],
        size_input_dim: int,
    ):
        super().__init__(non_zero_idcs, blocks, size_input_dim=size_input_dim)
        self.non_zero_idcs = torch.atleast_2d(non_zero_idcs)
        self.non_zero_idcs.requires_grad = False  # Ensure indices cannot be optimized
        self.blocks = torch.atleast_2d(blocks)
        self.size_input_dim = size_input_dim

    def _matmul(
        self: Float[LinearOperator, "*batch M N"],
        rhs: Float[torch.Tensor, "*batch2 N C"],
    ) -> Union[Float[torch.Tensor, "... M C"], Float[torch.Tensor, "... M"]]:
        # Workarounds for (Added)DiagLinearOperator
        # There seems to be a bug in DiagLinearOperator, which doesn't allow subsetting the way we do here.
        if isinstance(rhs, AddedDiagLinearOperator):
            return self._matmul(rhs._linear_op) + self._matmul(rhs._diag_tensor)

        if isinstance(rhs, DiagLinearOperator):
            return BlockDiagonalSparseLinearOperator(
                non_zero_idcs=self.non_zero_idcs,
                blocks=rhs.diag()[self.non_zero_idcs] * self.blocks,
                size_input_dim=self.size_input_dim,
            ).to_dense()

        # Subset rhs via index tensor
        rhs_non_zero = rhs[..., self.non_zero_idcs, :]

        if rhs.ndim == 2 and rhs.shape[-1] == 1:
            # TODO: Why is the below seemingly faster for (small-scale, i.e. 1e3 - 1e4) vectors?
            # Multiply and sum on sparse dimension
            return (self.blocks.unsqueeze(-1) * rhs_non_zero).sum(dim=-2)

        # Multiply on sparse dimension
        return (self.blocks.unsqueeze(-2) @ rhs_non_zero).squeeze(-2)

    def _size(self) -> torch.Size:
        return torch.Size((self.non_zero_idcs.shape[0], self.size_input_dim))

    def to_dense(self: LinearOperator) -> Tensor:
        if self.size() == self.blocks.shape:
            return self.blocks
        return torch.zeros(
            (self.blocks.shape[0], self.size_input_dim), dtype=self.blocks.dtype, device=self.blocks.device
        ).scatter_(src=self.blocks, index=self.non_zero_idcs, dim=1)
